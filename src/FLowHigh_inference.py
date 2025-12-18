import sys, os
# add project root to sys.path so 'models' package (models/__init__.py) is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import numpy as np
import librosa
import scipy
from torchinfo import summary
from models.FLowHigh.cfm_superresolution import (
    MelVoco,
    FLowHigh,
    ConditionalFlowMatcherWrapper
)
from models.FLowHigh.postprocessing import PostProcessing
from externals.ssr_eval import SSR_Eval_Helper 

class Complete_FLowHigh(object):

    """
    def __init__(self,
        input_sampling_rate,
        target_sampling_rate=48000,
        up_sampling_method='librosa',
        architecture='transformer',
        time_step=1,
        ode_method='midpoint',  # 'euler' is faster
        cfm_method='independent_cfm_adaptive',  # 'basic_cfm', 'independent_cfm_constant'
        sigma='0.0001',
        n_layers=2,
        n_heads=16,
        dim_head=64,
        n_mels=256,
        f_max=24000,
        n_fft=2048,
        win_length=2048,
        hop_length=480,
        vocoder='bigvgan',
        vocoder_config = 'models/FLowHigh/vocoder/BIGVGAN/config/bigvgan_48khz_256band_config.json',
        vocoder_path = 'models/FLowHigh/vocoder/BIGVGAN/checkpoint/g_48_00850000',
        model_path='models/FLowHigh/model/FLowHigh_indep_adaptive_400k/FLowHigh_indep_adaptive_400k.pt'
    ):
    """
    def __init__(self,
                model_config_path,
                model_path,
                input_sr,
                up_sampling_method='scipy',
                time_step=1,
                ode_method='midpoint',  # euler is faster
                vocoder_home='models/FLowHigh/vocoder/BIGVGAN'):
        with open(model_config_path) as f:
            properties = json.load(f)

        # for post-processing
        self.sr = input_sr
        self.post_processing = PostProcessing(0)
        self.up_sampling_method = up_sampling_method
        self.target_sampling_rate = properties['data']['samplingrate']
        self.cfm_method = properties['model']['cfm_path']
        self.timestep = time_step


        vocoder_path = '/'.join([vocoder_home, *properties['model']['vocoderpath'].split('/')[-2:]])
        vocoder_config = '/'.join([vocoder_home, *properties['model']['vocoderconfigpath'].split('/')[-2:]])

        print(f'Initializing FLowHigh...')
        audio_enc_dec_type_for_infer = MelVoco(n_mels= properties['data']['n_mel_channels'], 
                                               sampling_rate= self.target_sampling_rate, 
                                               f_max= properties['data']['mel_fmax'], 
                                               n_fft= properties['data']['n_fft'], 
                                               win_length= properties['data']['win_length'], 
                                               hop_length= properties['data']['hop_length'], 
                                               vocoder=properties['model']['vocoder'], 
                                               vocoder_config= vocoder_config, 
                                               vocoder_path = vocoder_path
                                                )  
              
        model_checkpoint = torch.load(model_path, map_location= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        SR_generator = FLowHigh(
            dim_in = audio_enc_dec_type_for_infer.n_mels, 
            audio_enc_dec = audio_enc_dec_type_for_infer,
            depth = properties['model']['n_layers'],
            dim_head = properties['model']['dim_head'],
            heads = properties['model']['n_heads'],
            architecture = properties['model']['architecture'])

        cfm_wrapper=ConditionalFlowMatcherWrapper(flowhigh=SR_generator, 
                                                cfm_method = properties['model']['cfm_path'], 
                                                torchdiffeq_ode_method=ode_method, 
                                                sigma = properties['model']['sigma'])
        cfm_wrapper.load_state_dict(model_checkpoint['model']) # dict_keys(['model', 'optim', 'scheduler'])
        SR_generator = SR_generator.cuda().eval()
        self.cfm_wrapper = cfm_wrapper.cuda().eval()

        number = sum(p.numel() for p in self.cfm_wrapper.parameters() if p.requires_grad)
        if number >= 1_000_000:
            print(f"Total number of parameters: {number / 1_000_000:.2f} million") 
        elif number >= 1_000:
            print(f"Total number of parameters: {number / 1_000:.2f} thousand") 
        else:
            print(f"Total number of parameters: {str(number)}")

        summary(cfm_wrapper)

    def up_sampling(self, audio):
        if self.up_sampling_method =='scipy':
            cond = scipy.signal.resample_poly(audio, self.target_sampling_rate, self.sr)
            cond /= np.max(np.abs(cond))
            if isinstance(cond, np.ndarray):
                cond = torch.tensor(cond).unsqueeze(0)
                cond = cond.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # [1, T]       
        elif self.up_sampling_method == 'librosa':
            cond = librosa.resample(audio, orig_sr=self.sr, target_sr=self.target_sampling_rate, res_type='soxr_hq')
            cond /= np.max(np.abs(cond))
            if isinstance(cond, np.ndarray):
                cond = torch.tensor(cond).unsqueeze(0)
            cond = cond.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # [1, T] 
        return cond

    def infer(self, audio):
        cond = self.up_sampling(audio)
        if self.cfm_method != 'independent_cfm_adaptive':
            HR_audio = self.cfm_wrapper.sample(cond = cond,
                                            time_steps=self.timestep,
                                            cfm_method=self.cfm_method)
        else:
            HR_audio = self.cfm_wrapper.sample(cond = cond,
                                            time_steps=self.timestep,
                                            cfm_method=self.cfm_method,
                                            std_2 = 1.)
        
        HR_audio_pp = self.post_processing.post_processing(HR_audio.squeeze(1), cond, cond.size(-1)) # [1, T] 
        # print(f"Input shape: {audio.shape} \n Post GAN shape: {HR_audio.shape} \n Post Process shape: {HR_audio_pp.shape}")
        return (HR_audio_pp.cpu().squeeze().clamp(-1,1).numpy()*32767.0).astype(np.float32)



def test():
    testee = Complete_FLowHigh(
                            model_config_path='models/FLowHigh/model/FLowHigh_indep_adaptive_400k/config.json',
                            model_path='models/FLowHigh/model/FLowHigh_indep_adaptive_400k/FLowHigh_indep_adaptive_400k.pt',
                            input_sr=44100,
                            ode_method='euler')
    # Initialize a evaluation helper
    helper = SSR_Eval_Helper(
        testee,
        test_name="FLowHigh_indep_adaptive_400k_euler_scipy",  # Test name for storing the result
        input_sr=44100,  # The sampling rate of the input x in the 'infer' function
        # output_sr=44100,  # The sampling rate of the output x in the 'infer' function
        output_sr=48000,
        evaluation_sr=48000,  # The sampling rate to calculate evaluation metrics.
        setting_fft={
            "cutoff_freq": [
                12000
            ],  # The cutoff frequency of the input x in the 'infer' function
        },
        save_processed_result=True
    )
    # Perform evaluation
    helper.evaluate(limit_test_nums=10, limit_test_speaker=-1)


test()
