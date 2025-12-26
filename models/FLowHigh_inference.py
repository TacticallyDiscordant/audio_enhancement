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

from models.rAI_FLowHigh import FlowHighSR


class Complete_FLowHigh(object):

    def __init__(self,
                model_config_path,
                model_path,
                input_sr,
                up_sampling_method='librosa',
                time_step=1,
                ode_method='midpoint',  # euler is faster
                use_vocoder=True,
                vocoder_home='models/FLowHigh/vocoder/BIGVGAN',
                vocoder_config=None,
                vocoder_path=None):
        with open(model_config_path) as f:
            properties = json.load(f)

        self.props = properties
        # for post-processing
        self.sr = input_sr
        self.post_processing = PostProcessing(0)
        self.up_sampling_method = up_sampling_method
        self.target_sampling_rate = self.props['data']['samplingrate']
        self.cfm_method = self.props['model']['cfm_path']
        self.timestep = time_step
        if use_vocoder:
            if vocoder_config is None:
                vocoder_config = '/'.join([vocoder_home, *self.props['model']['vocoderconfigpath'].split('/')[-2:]])
            if vocoder_path is None:
                vocoder_path = '/'.join([vocoder_home, *self.props['model']['vocoderpath'].split('/')[-2:]])

            print(f'Initializing FLowHigh...')
            audio_enc_dec_type_for_infer = MelVoco(n_mels=self.props['data']['n_mel_channels'], 
                                                   sampling_rate=self.target_sampling_rate, 
                                                   f_max=self.props['data']['mel_fmax'], 
                                                   n_fft=self.props['data']['n_fft'], 
                                                   win_length=self.props['data']['win_length'], 
                                                   hop_length=self.props['data']['hop_length'], 
                                                   vocoder=self.props['model']['vocoder'], 
                                                   vocoder_config=vocoder_config, 
                                                   vocoder_path=vocoder_path)  
              
        model_checkpoint = torch.load(model_path, map_location= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        
        SR_generator = FLowHigh(
                dim_in = audio_enc_dec_type_for_infer.n_mels, 
                audio_enc_dec = audio_enc_dec_type_for_infer,
                depth = self.props['model']['n_layers'],
                dim_head = self.props['model']['dim_head'],
                heads = self.props['model']['n_heads'],
                architecture = self.props['model']['architecture'])

        cfm_wrapper=ConditionalFlowMatcherWrapper(flowhigh=SR_generator, 
                                                cfm_method = self.props['model']['cfm_path'], 
                                                torchdiffeq_ode_method=ode_method, 
                                                sigma = self.props['model']['sigma'])
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
                                            cfm_method=self.cfm_method,
                                            mel_pp=False)
        else:
            HR_audio = self.cfm_wrapper.sample(cond = cond,
                                            time_steps=self.timestep,
                                            cfm_method=self.cfm_method,
                                            std_2 = 1.,
                                            mel_pp=False)
        
        HR_audio_pp = self.post_processing.post_processing(HR_audio.squeeze(1), cond, cond.size(-1)) # [1, T] 
        # print(f"Input shape: {audio.shape} \n Post GAN shape: {HR_audio.shape} \n Post Process shape: {HR_audio_pp.shape}")
        return (HR_audio_pp.cpu().squeeze().clamp(-1,1).numpy()*32767.0).astype(np.float32)

class rAI_FLowHigh(object):
        def __init__(self,
                        input_sr,
                        target_sr):
            self.input_sr = input_sr
            self.target_sr = target_sr
            self.model = FlowHighSR.from_pretrained(device="cuda")

        def infer(self, audio):
            prediction = self.model.generate(audio=audio,
                                        sr=self.input_sr,
                                        target_sampling_rate=self.target_sr)
            return prediction.cpu().squeeze().numpy()
