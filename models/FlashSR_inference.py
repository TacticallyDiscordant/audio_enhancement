import librosa
import torch
import time
import onnxruntime as ort
import numpy as np
import soundfile as sf
from models.FastAudioSR import FASR
from huggingface_hub import hf_hub_download


class FlashSR_ort():

    def __init__(self, model_path,
                    input_sr,
                    target_sr):
        self.input_sr = input_sr
        self.output_sr = target_sr
        # Create ONNX session and run inference
        self.session = ort.InferenceSession(model_path)
        
        # Print model input/output info
        print("\n=== ONNX Model Info ===")
        print("Inputs:")
        for inp in self.session.get_inputs():
            print(f"  Name: '{inp.name}', Shape: {inp.shape}, Type: {inp.type}")
        print("Outputs:")
        for out in self.session.get_outputs():
            print(f"  Name: '{out.name}', Shape: {out.shape}, Type: {out.type}")
        print("=======================\n")
    
    def timed_infer(self, audio):
        if audio.ndim == 1:
            audio = audio.reshape(1, 1, -1)
        elif audio.ndim == 2:
            # If audio is 2D (channels, samples), add batch dimension
            audio = audio.reshape(1, audio.shape[0], audio.shape[1])
        
        # Convert to float32 if needed
        audio = audio.astype(np.float32)
        
        t1 = time.time()
        # Use the actual input/output names from the model
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        output = self.session.run([output_name], {input_name: audio})[0]
        t2 = time.time()
        
        # Squeeze output to remove batch and channel dimensions [1, 1, samples] -> [samples]
        output = output.squeeze()
        
        # Model outputs at 3x input SR, resample to target SR
        model_output_sr = self.input_sr * 3  # 44100 * 3 = 132300 Hz
        if model_output_sr != self.output_sr:
            output = librosa.resample(output, orig_sr=model_output_sr, target_sr=self.output_sr)
        
        inference_time = t2-t1
        inference_time_per_second = inference_time / (audio.shape[-1]/self.input_sr)
        return output, {'inference_speed': inference_time_per_second}

class FlashSR():

    def __init__(self, model_path,
                    input_sr,
                    target_sr):
        self.input_sr = input_sr
        self.output_sr = target_sr

        self.model = FASR(model_path)


    def timed_infer(self, audio):
        # Convert numpy array to torch tensor
        audio = torch.from_numpy(audio).float()
        
        t1 = time.time()
        prediction = self.model.run(audio)
        t2 = time.time()
        
        # Convert back to numpy and squeeze
        output = prediction.cpu().numpy().squeeze()
        
        # Model outputs at 3x input SR, resample to target SR if needed
        model_output_sr = self.input_sr * 3  # 44100 * 3 = 132300 Hz
        if model_output_sr != self.output_sr:
            output = librosa.resample(output, orig_sr=model_output_sr, target_sr=self.output_sr)
        
        inference_time = t2-t1
        inference_time_per_second = inference_time / (len(audio)/self.input_sr)
        return output, {'inference_speed': inference_time_per_second}