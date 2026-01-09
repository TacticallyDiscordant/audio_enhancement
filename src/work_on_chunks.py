import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib import style
from pathlib import Path
import argparse
from models.FLowHigh_inference import rAI_FLowHigh
import models.FlashSR_inference as flash


style.use('fivethirtyeight')
style.use('dark_background')

def compute_log_spectral_distance(audio1, audio2, sr, n_fft=2048, hop_length=512):
    """Compute log-spectral distance between two audio signals"""
    # Compute spectrograms
    spec1 = np.abs(librosa.stft(audio1, n_fft=n_fft, hop_length=hop_length))
    spec2 = np.abs(librosa.stft(audio2, n_fft=n_fft, hop_length=hop_length))
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    log_spec1 = np.log10(spec1 + eps)
    log_spec2 = np.log10(spec2 + eps)
    
    # Compute LSD
    lsd = np.sqrt(np.mean((log_spec1 - log_spec2) ** 2))
    return lsd


def compute_mel_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Compute mel-spectrogram for visualization"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def process_audio_in_chunks(input_path, output_path, model, chunk_size_seconds=5.0, overlap_seconds=0.5):
    """
    Process audio file in chunks and collect metrics
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save enhanced audio
        model: Model with timed_infer method
        chunk_size_seconds: Length of each chunk in seconds
        overlap_seconds: Overlap between chunks for smooth transitions
    """
    # Load audio
    audio, sr = librosa.load(input_path, sr=model.input_sr, mono=True)
    print(f"Loaded audio: {len(audio)/sr:.2f}s @ {sr}Hz")
    
    # Calculate chunk parameters
    chunk_size = int(chunk_size_seconds * sr)
    overlap_size = int(overlap_seconds * sr)
    hop_size = chunk_size - overlap_size
    
    # Storage for results
    enhanced_chunks = []
    original_chunks = []
    lsd_scores = []
    inference_times = []
    
    # Process chunks
    num_chunks = int(np.ceil((len(audio) - overlap_size) / hop_size))
    print(f"Processing {num_chunks} chunks...")
    
    for i in range(num_chunks):
        start_idx = i * hop_size
        end_idx = min(start_idx + chunk_size, len(audio))
        
        # Extract chunk
        chunk = audio[start_idx:end_idx]
        
        # Pad last chunk if needed
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
        
        # Process chunk
        enhanced_chunk, metrics = model.timed_infer(chunk)
        
        # Store results
        original_chunks.append(chunk)
        enhanced_chunks.append(enhanced_chunk)
        inference_times.append(metrics['inference_speed'])
        
        # Compute LSD (upsample original to target_sr for comparison)
        chunk_upsampled = librosa.resample(chunk, orig_sr=sr, target_sr=model.target_sr)
        lsd = compute_log_spectral_distance(
            chunk_upsampled[:len(enhanced_chunk)], 
            enhanced_chunk, 
            model.target_sr
        )
        lsd_scores.append(lsd)
        
        print(f"Chunk {i+1}/{num_chunks}: LSD={lsd:.4f}, Speed={metrics['inference_speed']:.4f}x")
    
    # Reconstruct full audio with overlap-add
    total_length = (num_chunks - 1) * hop_size + chunk_size
    total_length_target = int(total_length * model.target_sr / sr)
    reconstructed = np.zeros(total_length_target)
    window = np.hanning(overlap_size * 2)
    
    for i, chunk in enumerate(enhanced_chunks):
        start_idx_target = int(i * hop_size * model.target_sr / sr)
        
        if i == 0:
            # First chunk: no fade-in
            reconstructed[start_idx_target:start_idx_target + len(chunk)] = chunk
        elif i == len(enhanced_chunks) - 1:
            # Last chunk: fade-in only
            fade_length = min(overlap_size * 2, len(chunk))
            fade = window[:fade_length]
            reconstructed[start_idx_target:start_idx_target + fade_length] *= (1 - fade)
            reconstructed[start_idx_target:start_idx_target + len(chunk)] += chunk * np.concatenate([fade, np.ones(len(chunk) - fade_length)])
        else:
            # Middle chunks: crossfade
            fade_length = overlap_size * 2
            fade = window
            reconstructed[start_idx_target:start_idx_target + fade_length] *= (1 - fade)
            reconstructed[start_idx_target:start_idx_target + len(chunk)] += chunk * np.concatenate([fade, np.ones(len(chunk) - fade_length)])
    
    # Trim to original length
    original_length_target = int(len(audio) * model.target_sr / sr)
    reconstructed = reconstructed[:original_length_target]
    
    # Save enhanced audio
    sf.write(output_path, reconstructed, model.target_sr)
    print(f"Saved enhanced audio to: {output_path}")
    
    return {
        'original_audio': audio,
        'enhanced_audio': reconstructed,
        'original_sr': sr,
        'target_sr': model.target_sr,
        'lsd_scores': lsd_scores,
        'inference_times': inference_times,
        'num_chunks': num_chunks
    }


def plot_results(results, output_plot_path):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Original Mel-Spectrogram
    mel_orig = compute_mel_spectrogram(results['original_audio'], results['original_sr'])
    im1 = axes[0, 0].imshow(mel_orig, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title(f'Original Mel-Spectrogram ({results["original_sr"]}Hz)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Mel Frequency')
    plt.colorbar(im1, ax=axes[0, 0], format='%+2.0f dB')
    
    # 2. Enhanced Mel-Spectrogram
    mel_enhanced = compute_mel_spectrogram(results['enhanced_audio'], results['target_sr'])
    im2 = axes[0, 1].imshow(mel_enhanced, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 1].set_title(f'Enhanced Mel-Spectrogram ({results["target_sr"]}Hz)')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Mel Frequency')
    plt.colorbar(im2, ax=axes[0, 1], format='%+2.0f dB')
    
    # 3. Log-Spectral Distance per chunk
    axes[1, 0].plot(results['lsd_scores'], marker='o', linestyle='-', linewidth=2)
    axes[1, 0].set_title('Log-Spectral Distance per Chunk')
    axes[1, 0].set_xlabel('Chunk Index')
    axes[1, 0].set_ylabel('LSD')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(np.mean(results['lsd_scores']), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(results["lsd_scores"]):.4f}')
    axes[1, 0].legend()
    
    # 4. Inference Time per chunk
    axes[1, 1].plot(results['inference_times'], marker='s', linestyle='-', linewidth=2, color='orange')
    axes[1, 1].set_title('Inference Speed per Chunk')
    axes[1, 1].set_xlabel('Chunk Index')
    axes[1, 1].set_ylabel('Inference Time / Audio Duration')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(np.mean(results['inference_times']), color='r', linestyle='--',
                       label=f'Mean: {np.mean(results["inference_times"]):.4f}x')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved plots to: {output_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Process audio in chunks with FLowHigh model')
    parser.add_argument('--input', type=str, required=True, help='Input audio file path')
    parser.add_argument('--output', type=str, default=None, help='Output audio file path')
    parser.add_argument('--plot', type=str, default=None, help='Output plot path')
    parser.add_argument('--input_sr', type=int, default=48000, help='Input sample rate')
    parser.add_argument('--target_sr', type=int, default=48000, help='Target sample rate')
    parser.add_argument('--chunk_size', type=float, default=.2, help='Chunk size in seconds')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap in seconds')
    
    args = parser.parse_args()
    
    # Set default output paths
    input_path = Path(args.input)
    if args.output is None:
        args.output = input_path.parent / f"{input_path.stem}_enhanced.wav"
    if args.plot is None:
        args.plot = input_path.parent / f"{input_path.stem}_analysis.png"
    
    # Initialize model
    print("Initializing model...")
    
    model = rAI_FLowHigh(
        input_sr=args.input_sr,
        target_sr=args.target_sr,
        basic=True,
        live_mode=False
    )
    
    """
    model = flash.FlashSR(model_path='./models/weights_and_configs/FlashSR/upsampler.pth',
                                input_sr=args.input_sr,
                                target_sr=args.target_sr
                                )
    """
    # Process audio
    results = process_audio_in_chunks(
        args.input,
        args.output,
        model,
        chunk_size_seconds=args.chunk_size,
        overlap_seconds=args.overlap
    )
    
    # Create plots
    plot_results(results, args.plot)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total chunks: {results['num_chunks']}")
    print(f"Mean LSD: {np.mean(results['lsd_scores']):.4f} ± {np.std(results['lsd_scores']):.4f}")
    print(f"Mean inference speed: {np.mean(results['inference_times']):.4f}x ± {np.std(results['inference_times']):.4f}x")


if __name__ == '__main__':
    main()