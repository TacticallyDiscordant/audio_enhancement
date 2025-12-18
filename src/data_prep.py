import os
from pathlib import Path
from beartype import beartype
import torch
from torch.utils.data import Dataset
import librosa
from scipy.signal import cheby1, sosfiltfilt
import numpy as np
import random
# import glob
import argparse
import soundfile as sf
from tqdm import trange

class AudioDataset(Dataset):
    """
    from FLow High,
    reduced to Librosa
    """
    @beartype
    def __init__(
        self,
        folder: str,
        target:str,
        audio_extension: str = ".flac",
        mode: str = None,
        audio_out_extension: str = ".wav"
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'
        self.output_dir = Path(target)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.audio_extension = audio_extension
        self.mode = mode
        files = list(path.glob(f'**/*{audio_extension}'))
        assert len(files) > 0, 'no files found'
        self.files = files


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        file = self.files[idx]

        wave, sr = librosa.load(file, sr=None, mono=True) # [Time] 
        wave /= np.max(np.abs(wave))
        nyq = sr // 2
        min_value = 4000
        max_value = 32000
        step = 1000
        sampling_rates = list(range(min_value, max_value + step, step))
        random_sr = random.choice(sampling_rates)
        
        if self.mode == 'valid':
            order = 8
            ripple = 0.05
        else:
            order = random.randint(1, 11)
            ripple = random.choice([1e-9, 1e-6, 1e-3, 1, 5])

        highcut = random_sr // 2
        hi = highcut / nyq
        sos = cheby1(order, ripple, hi, btype='lowpass', output='sos')
        d_HR_wave = sosfiltfilt(sos, wave)
        down_cond = librosa.resample(d_HR_wave, orig_sr=sr, target_sr=random_sr, res_type='soxr_hq')
        up_cond = librosa.resample(down_cond, orig_sr=random_sr, target_sr=sr, res_type='soxr_hq')

        if len(up_cond) < len(wave):
            up_cond = np.pad(wave, (0, len(wave) - len(up_cond)), 'constant', constant_values=0)
        elif len(up_cond) > len(wave):
            up_cond = up_cond[:len(wave)]
        
        # length = wave.shape[-1]

        # if self.mode == 'valid':
        #     return torch.from_numpy(wave).float(), length
        return up_cond, random_sr


    def write_file(self, idx, ): 
        """
        use soundfile to write Ground Truth and degraded files with random samplerate
        """
        
        file = self.files[idx]
        wave, sr = self[idx]
        # subtype = 'PCM_16'
        file_name = file.stem + '.flac'
        dir_name = file.parts[-2]
        target_path = self.output_dir / 'degraded' / dir_name
        gt_path = self.output_dir / 'gt' / dir_name
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)
        
        # reload ground truth
        gt, gt_sr = librosa.load(file, sr=None, mono=True)

        # write ground truth and degraded files as '.wav' into their respective directories
        # due to lack of HD space, I can't create a same type ground truth
        sf.write(gt_path/file_name, data=gt, samplerate=gt_sr, format="FLAC")
        sf.write(target_path/file_name, data=wave, samplerate=sr, format="FLAC")
        # print(f"Written {file_name} to {target_path/file_name}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate degrate dataset")
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    ad = AudioDataset(folder=args.input_path, target=args.output_path)
    for idx in trange(len(ad.files)):
        # up_cond, random_sr = ad[idx]
        ad.write_file(idx)

    
