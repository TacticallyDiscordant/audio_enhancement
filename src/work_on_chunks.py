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
import scipy.signal

style.use('fivethirtyeight')
style.use('dark_background')

def compute_log_spectral_distance(audio1, audio2, sr, n_fft=2048, hop_length=512):
    """Compute log-spectral distance between two audio signals"""
    # Ensure same length
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
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


def process_audio_in_chunks(input_path, output_path, model, ground_truth_path=None, chunk_size_seconds=5.0, overlap_seconds=0.5):
    """
    Process audio file in chunks and collect metrics
    
    Args:
        input_path: Path to degraded audio file
        output_path: Path to save enhanced audio
        model: Model with timed_infer method
        ground_truth_path: Path to ground truth audio file (optional)
        chunk_size_seconds: Length of each chunk in seconds
        overlap_seconds: Overlap between chunks for smooth transitions
    """
    # Load audio
    audio, sr = librosa.load(input_path, sr=model.input_sr, mono=True)
    print(f"Loaded audio: {len(audio)/sr:.2f}s @ {sr}Hz")
    
    # Load ground truth if provided
    ground_truth = None
    ground_truth_sr = None
    if ground_truth_path:
        ground_truth, ground_truth_sr = librosa.load(ground_truth_path, sr=model.target_sr, mono=True)
        print(f"Loaded ground truth: {len(ground_truth)/ground_truth_sr:.2f}s @ {ground_truth_sr}Hz")
    
    # Calculate chunk parameters
    chunk_size = int(chunk_size_seconds * sr)
    overlap_size = int(overlap_seconds * sr)
    hop_size = chunk_size - overlap_size
    
    # Add extra context padding (process more, use less)
    context_size = overlap_size  # Extra samples on each side for context
    
    # Storage for results
    enhanced_chunks = []
    degraded_chunks = []
    lsd_degraded_to_gt = []
    lsd_enhanced_to_gt = []
    inference_times = []
    
    # Process chunks
    num_chunks = int(np.ceil((len(audio) - overlap_size) / hop_size))
    print(f"Processing {num_chunks} chunks with context...")
    
    for i in range(num_chunks):
        start_idx = i * hop_size
        end_idx = min(start_idx + chunk_size, len(audio))
        
        # Add context on both sides
        context_start = max(0, start_idx - context_size)
        context_end = min(len(audio), end_idx + context_size)
        
        # Extract chunk with context
        chunk_with_context = audio[context_start:context_end]
        
        # Pad if needed
        target_length = chunk_size + 2 * context_size
        if len(chunk_with_context) < target_length:
            chunk_with_context = np.pad(
                chunk_with_context, 
                (0, target_length - len(chunk_with_context)), 
                mode='constant'
            )
        
        # Process chunk with context
        enhanced_with_context, metrics = model.timed_infer(chunk_with_context)
        
        # Calculate how much context was actually added
        left_context = start_idx - context_start
        right_context = context_end - end_idx
        
        # Extract only the valid center portion (discard context edges)
        left_context_target = int(left_context * model.target_sr / sr)
        chunk_length_target = int((end_idx - start_idx) * model.target_sr / sr)
        
        enhanced_chunk = enhanced_with_context[
            left_context_target:left_context_target + chunk_length_target
        ]
        
        # Store chunk (without context)
        chunk = audio[start_idx:end_idx]
        degraded_chunks.append(chunk)
        enhanced_chunks.append(enhanced_chunk)
        inference_times.append(metrics['inference_speed'])
        
        # Compute LSD against ground truth if available
        if ground_truth is not None:
            gt_start_idx = int(start_idx * model.target_sr / sr)
            gt_end_idx = gt_start_idx + len(enhanced_chunk)
            
            if gt_end_idx <= len(ground_truth):
                gt_chunk = ground_truth[gt_start_idx:gt_end_idx]
                
                # LSD: Degraded to Ground Truth (upsample degraded first)
                chunk_upsampled = librosa.resample(chunk, orig_sr=sr, target_sr=model.target_sr)
                lsd_deg_gt = compute_log_spectral_distance(
                    chunk_upsampled[:len(gt_chunk)], 
                    gt_chunk, 
                    model.target_sr
                )
                lsd_degraded_to_gt.append(lsd_deg_gt)
                
                # LSD: Enhanced to Ground Truth
                lsd_enh_gt = compute_log_spectral_distance(
                    enhanced_chunk[:len(gt_chunk)], 
                    gt_chunk, 
                    model.target_sr
                )
                lsd_enhanced_to_gt.append(lsd_enh_gt)
                
                print(f"Chunk {i+1}/{num_chunks}: LSD_deg→GT={lsd_deg_gt:.4f}, LSD_enh→GT={lsd_enh_gt:.4f}, Speed={metrics['inference_speed']:.4f}x")
            else:
                print(f"Chunk {i+1}/{num_chunks}: Beyond GT length, Speed={metrics['inference_speed']:.4f}x")
        else:
            print(f"Chunk {i+1}/{num_chunks}: Speed={metrics['inference_speed']:.4f}x")
    
    # Reconstruct full audio with improved overlap-add using Tukey window
    total_length = (num_chunks - 1) * hop_size + chunk_size
    total_length_target = int(total_length * model.target_sr / sr)
    hop_size_target = int(hop_size * model.target_sr / sr)
    overlap_size_target = int(overlap_size * model.target_sr / sr)
    
    reconstructed = np.zeros(total_length_target)
    window_sum = np.zeros(total_length_target)  # Track normalization
    
    # Create Tukey (tapered cosine) window for smoother transitions
    if overlap_size_target > 0:
        # alpha controls the taper: 0=rectangular, 1=Hann
        alpha = 1.0  # Full cosine taper
        tukey_window = scipy.signal.windows.tukey(overlap_size_target * 2, alpha=alpha)
        fade_out = tukey_window[:overlap_size_target]
        fade_in = tukey_window[overlap_size_target:]
    
    for i, chunk in enumerate(enhanced_chunks):
        start_idx_target = i * hop_size_target
        chunk_length = len(chunk)
        end_idx_target = start_idx_target + chunk_length
        
        # Create window for this chunk
        window = np.ones(chunk_length)
        
        if overlap_size_target > 0:
            # Apply fade-in at the start (except first chunk)
            if i > 0:
                fade_length = min(overlap_size_target, chunk_length)
                window[:fade_length] = fade_in[:fade_length]
            
            # Apply fade-out at the end (except last chunk)
            if i < len(enhanced_chunks) - 1:
                fade_length = min(overlap_size_target, chunk_length)
                window[-fade_length:] = fade_out[:fade_length]
        
        # Add windowed chunk
        valid_end = min(end_idx_target, len(reconstructed))
        valid_length = valid_end - start_idx_target
        reconstructed[start_idx_target:valid_end] += chunk[:valid_length] * window[:valid_length]
        window_sum[start_idx_target:valid_end] += window[:valid_length]
    
    # Normalize by window sum to avoid amplitude changes
    window_sum[window_sum < 1e-8] = 1.0  # Avoid division by zero
    reconstructed /= window_sum
    
    # Trim to original length
    original_length_target = int(len(audio) * model.target_sr / sr)
    reconstructed = reconstructed[:original_length_target]
    
    # Save enhanced audio
    sf.write(output_path, reconstructed, model.target_sr)
    print(f"Saved enhanced audio to: {output_path}")
    
    return {
        'degraded_audio': audio,
        'enhanced_audio': reconstructed,
        'ground_truth_audio': ground_truth,
        'degraded_sr': sr,
        'target_sr': model.target_sr,
        'ground_truth_sr': ground_truth_sr,
        'lsd_degraded_to_gt': lsd_degraded_to_gt,
        'lsd_enhanced_to_gt': lsd_enhanced_to_gt,
        'inference_times': inference_times,
        'num_chunks': num_chunks
    }


def plot_results(results, output_plot_path):
    """Create visualization plots"""
    has_gt = results['ground_truth_audio'] is not None
    
    if has_gt:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Degraded Mel-Spectrogram
    mel_degraded = compute_mel_spectrogram(results['degraded_audio'], results['degraded_sr'])
    im1 = axes[0, 0].imshow(mel_degraded, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title('Degraded')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Mel Frequency')
    axes[0, 0].grid(False)
    plt.colorbar(im1, ax=axes[0, 0], format='%+2.0f dB')
    
    # 2. Enhanced Mel-Spectrogram
    mel_enhanced = compute_mel_spectrogram(results['enhanced_audio'], results['target_sr'])
    im2 = axes[0, 2].imshow(mel_enhanced, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 2].set_title('Enhanced')
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('Mel Frequency')
    axes[0, 2].grid(False)
    plt.colorbar(im2, ax=axes[0, 2], format='%+2.0f dB')
    
    if has_gt:
        # 3. Ground Truth Mel-Spectrogram
        mel_gt = compute_mel_spectrogram(results['ground_truth_audio'], results['ground_truth_sr'])
        im3 = axes[0, 1].imshow(mel_gt, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Mel Frequency')
        axes[0, 1].grid(False)
        plt.colorbar(im3, ax=axes[0, 1], format='%+2.0f dB')
        
        # 4. LSD Comparison
        x = range(len(results['lsd_degraded_to_gt']))
        axes[1, 0].plot(x, results['lsd_degraded_to_gt'], color='steelblue', marker='o', linestyle='-', linewidth=2, label='Degraded → GT')
        axes[1, 0].plot(x, results['lsd_enhanced_to_gt'], color='coral', marker='s', linestyle='-', linewidth=2, label='Enhanced → GT')
        axes[1, 0].set_title('Log-Spectral Distance')
        axes[1, 0].set_xlabel('Chunk Index')
        axes[1, 0].set_ylabel('LSD')
        axes[1, 0].grid(True, alpha=0.3)
        # axes[1, 0].axhline(np.mean(results['lsd_degraded_to_gt']), color='steelblue', linestyle='--', alpha=0.7,
        #                   label=f'Mean Deg→GT: {np.mean(results["lsd_degraded_to_gt"]):.4f}')
        # axes[1, 0].axhline(np.mean(results['lsd_enhanced_to_gt']), color='coral', linestyle='--', alpha=0.7,
        #                   label=f'Mean Enh→GT: {np.mean(results["lsd_enhanced_to_gt"]):.4f}')
        axes[1, 0].legend()
        
        # 5. Inference Time
        axes[1, 2].plot(results['inference_times'][1:], marker='s', linestyle='-', linewidth=2, color='orange')
        axes[1, 2].set_title('Inference Speed')
        axes[1, 2].set_xlabel('Chunk Index')
        axes[1, 2].set_ylabel('Inference Time / Audio Duration')
        axes[1, 2].grid(True, alpha=0.3)
        # axes[1, 2].axhline(np.mean(results['inference_times']), color='r', linestyle='--',
        #                  label=f'Mean: {np.mean(results["inference_times"]):.4f}x')
        # axes[1, 2].legend()
        
        # 6. Empty placeholder
        axes[1, 1].axis('off')
        
    else:
        # Without ground truth: show original plots
        # 3. Inference Time
        axes[1, 0].plot(results['inference_times'], marker='s', linestyle='-', linewidth=2, color='orange')
        axes[1, 0].set_title('Inference Speed per Chunk')
        axes[1, 0].set_xlabel('Chunk Index')
        axes[1, 0].set_ylabel('Inference Time / Audio Duration')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(np.mean(results['inference_times']), color='r', linestyle='--',
                          label=f'Mean: {np.mean(results["inference_times"]):.4f}x')
        axes[1, 0].legend()
        
        # 4. Empty placeholder
        axes[1, 1].text(0.5, 0.5, 'No Ground Truth Provided', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Saved plots to: {output_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Process audio in chunks with FLowHigh model')
    parser.add_argument('--input', type=str, required=True, help='Degraded audio file path')
    parser.add_argument('--ground_truth', type=str, default=None, help='Ground truth audio file path')
    parser.add_argument('--output', type=str, default=None, help='Output audio file path')
    parser.add_argument('--plot', type=str, default=None, help='Output plot path')
    parser.add_argument('--input_sr', type=int, default=48000, help='Degraded audio sample rate')
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
    """
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
    
    # Process audio
    results = process_audio_in_chunks(
        args.input,
        args.output,
        model,
        ground_truth_path=args.ground_truth,
        chunk_size_seconds=args.chunk_size,
        overlap_seconds=args.overlap
    )
    
    # Create plots
    plot_results(results, args.plot)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total chunks: {results['num_chunks']}")
    if results['ground_truth_audio'] is not None:
        print(f"Mean LSD (Degraded → GT): {np.mean(results['lsd_degraded_to_gt']):.4f} ± {np.std(results['lsd_degraded_to_gt']):.4f}")
        print(f"Mean LSD (Enhanced → GT): {np.mean(results['lsd_enhanced_to_gt']):.4f} ± {np.std(results['lsd_enhanced_to_gt']):.4f}")
        improvement = np.mean(results['lsd_degraded_to_gt']) - np.mean(results['lsd_enhanced_to_gt'])
        print(f"Mean LSD Improvement: {improvement:.4f} ({'better' if improvement > 0 else 'worse'})")
    print(f"Mean inference speed: {np.mean(results['inference_times']):.4f}x ± {np.std(results['inference_times']):.4f}x")


if __name__ == '__main__':
    main()