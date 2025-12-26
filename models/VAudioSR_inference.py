import gc
import os
import random
import numpy as np
from scipy.signal.windows import hann
import soundfile as sf
import torch
from cog import BasePredictor, Input, Path
import tempfile
import argparse
import librosa
from models.audiosr.pipeline import build_model, super_resolution_from_wave
from scipy import signal
import pyloudnorm as pyln


import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")

def match_array_shapes(array_1:np.ndarray, array_2:np.ndarray):
    if (len(array_1.shape) == 1) & (len(array_2.shape) == 1):
        if array_1.shape[0] > array_2.shape[0]:
            array_1 = array_1[:array_2.shape[0]]
        elif array_1.shape[0] < array_2.shape[0]:
            array_1 = np.pad(array_1, ((array_2.shape[0] - array_1.shape[0], 0)), 'constant', constant_values=0)
    else:
        if array_1.shape[1] > array_2.shape[1]:
            array_1 = array_1[:,:array_2.shape[1]]
        elif array_1.shape[1] < array_2.shape[1]:
            padding = array_2.shape[1] - array_1.shape[1]
            array_1 = np.pad(array_1, ((0,0), (0,padding)), 'constant', constant_values=0)
    return array_1


def lr_filter(audio, cutoff, filter_type, order=12, sr=48000):
    audio = audio.T
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order//2, normal_cutoff, btype=filter_type, analog=False)
    sos = signal.tf2sos(b, a)
    filtered_audio = signal.sosfiltfilt(sos, audio)
    return filtered_audio.T


class Predictor(BasePredictor):
    def setup(self,
                model_name="basic",
                device="auto",
                input_sr=48000,
                output_sr=48000,
                chunk_size=5.12,
                overlap=0.1,
                guidance_scale=3.5,
                ddim_steps=50):
        self.model_name = model_name
        self.device = device
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.guidance_scale = guidance_scale
        self.ddim_steps = ddim_steps
        print("Loading Model...")
        self.audiosr = build_model(model_name=self.model_name, device=self.device)
        # print(self.audiosr)
        # exit()
        print("Model loaded!")

    def process_audio(self, audio):
        audio = audio.T
        # print(f"audio.shape = {audio.shape}")
        # print(f"input cutoff = {input_cutoff}")
        
        is_stereo = len(audio.shape) == 2
        audio_channels = [audio] if not is_stereo else [audio[:, 0], audio[:, 1]]
        # print("audio is stereo" if is_stereo else "Audio is mono")

        chunk_samples = int(self.chunk_size * self.input_sr)
        overlap_samples = int(self.overlap * chunk_samples)
        output_chunk_samples = int(self.chunk_size * self.output_sr)
        output_overlap_samples = int(self.overlap * output_chunk_samples)
        enable_overlap = self.overlap > 0
        # print(f"enable_overlap = {enable_overlap}")
        
        def process_chunks(audio):
            chunks = []
            original_lengths = []
            start = 0
            while start < len(audio):
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]
                if len(chunk) < chunk_samples:
                    original_lengths.append(len(chunk))
                    chunk = np.concatenate([chunk, np.zeros(chunk_samples - len(chunk))])
                else:
                    original_lengths.append(chunk_samples)
                chunks.append(chunk)
                start += chunk_samples - overlap_samples if enable_overlap else chunk_samples
            return chunks, original_lengths

        # Process both channels (mono or stereo)
        chunks_per_channel = [process_chunks(channel) for channel in audio_channels]
        sample_rate_ratio =  self.output_sr / self.input_sr
        # total_length = len(chunks_per_channel[0][0]) * output_chunk_samples - (len(chunks_per_channel[0][0]) - 1) * (output_overlap_samples if enable_overlap else 0)
        orig_samples = len(audio_channels[0])
        total_length = int(orig_samples * sample_rate_ratio)
        reconstructed_channels = [np.zeros(total_length, dtype=np.float32) for _ in audio_channels]

        meter_before = pyln.Meter(self.input_sr)
        meter_after = pyln.Meter(self.output_sr)
        
        def safe_integrated_loudness(meter, arr: np.ndarray):
            a = np.asarray(arr).flatten()
            # pyloudnorm requires length strictly greater than block_size
            min_len = int(meter.block_size) + 1
            if a.size < min_len:
                return None
            # ensure dtype is float
            a = a.astype(np.float32)
            try:
                return meter.integrated_loudness(a)
            except Exception:
                return None

        # Process chunks for each channel
        for ch_idx, (chunks, original_lengths) in enumerate(chunks_per_channel):
            for i, chunk in enumerate(chunks):
                loudness_before = safe_integrated_loudness(meter_before, chunk)
                print(f"Processing chunk {i+1} of {len(chunks)} for {'Left/Mono' if ch_idx == 0 else 'Right'} channel")
                out_chunk = super_resolution_from_wave(
                        self.audiosr,
                        chunk,
                        guidance_scale=self.guidance_scale,
                        ddim_steps=self.ddim_steps,
                        latent_t_per_second=12.8
                        )

                out_chunk = out_chunk[0]
                num_samples_to_keep = int(original_lengths[i] * sample_rate_ratio)
                out_chunk = out_chunk[:, :num_samples_to_keep].squeeze()
                loudness_after = safe_integrated_loudness(meter_after, out_chunk)
                # only normalize when both loudness values are available and valid
                if (loudness_before is not None) and (loudness_after is not None):
                    out_chunk = pyln.normalize.loudness(out_chunk, loudness_after, loudness_before)

                if enable_overlap:
                    actual_overlap_samples = min(output_overlap_samples, num_samples_to_keep)
                    fade_out = np.linspace(1., 0., actual_overlap_samples)
                    fade_in = np.linspace(0., 1., actual_overlap_samples)

                    if i == 0:
                        out_chunk[-actual_overlap_samples:] *= fade_out
                    elif i < len(chunks) - 1:
                        out_chunk[:actual_overlap_samples] *= fade_in
                        out_chunk[-actual_overlap_samples:] *= fade_out
                    else:
                        out_chunk[:actual_overlap_samples] *= fade_in

                start = i * (output_chunk_samples - output_overlap_samples if enable_overlap else output_chunk_samples)
                if start >= total_length:
                    continue
                end = min(start + out_chunk.shape[0], total_length)
                needed = end - start
                chunk_to_add = out_chunk.flatten()[:needed]
                if chunk_to_add.shape[0] < needed:
                    chunk_to_add = np.pad(chunk_to_add, (0, needed - chunk_to_add.shape[0]), mode='constant')
                reconstructed_channels[ch_idx][start:end] += chunk_to_add


        reconstructed_audio = np.stack(reconstructed_channels, axis=-1) if is_stereo else reconstructed_channels[0]
        # print(output, type(output))
        # return reconstructed_audio[0]
        return reconstructed_audio

    def infer(self,
        audio: list = Input(description="Audio to upsample")):
        waveform = self.process_audio(audio)
        return waveform.squeeze()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Find volume difference of two audio files.")
    parser.add_argument("--input", help="Path to input audio file")
    parser.add_argument("--output", help="Output folder")
    parser.add_argument("--ddim_steps", help="Number of ddim steps", type=int, required=False, default=50)
    parser.add_argument("--chunk_size", help="chunk size", type=float, required=False, default=10.24)
    parser.add_argument("--guidance_scale", help="Guidance scale value",  type=float, required=False, default=3.5)
    parser.add_argument("--seed", help="Seed value, 0 = random seed", type=int, required=False, default=0)
    parser.add_argument("--overlap", help="overlap value", type=float, required=False, default=0.04)
    parser.add_argument("--multiband_ensemble", type=bool, help="Use multiband ensemble with input")
    parser.add_argument("--input_cutoff", help="Define the crossover of audio input in the multiband ensemble", type=int, required=False, default=12000)

    args = parser.parse_args()

    input_file_path = args.input
    output_folder = args.output
    ddim_steps = args.ddim_steps
    chunk_size = args.chunk_size
    guidance_scale = args.guidance_scale
    seed = args.seed
    overlap = args.overlap
    input_cutoff = args.input_cutoff
    multiband_ensemble = args.multiband_ensemble

    crossover_freq = input_cutoff - 1000

    p = Predictor()
    
    p.setup(device='auto')


    out = p.predict(
        input_file_path,
        ddim_steps=ddim_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        chunk_size=chunk_size,
        overlap=overlap
    )

    del p
    gc.collect()
    torch.cuda.empty_cache()