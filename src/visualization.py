import os
import matplotlib
matplotlib.use('TkAgg')  # MUST be before any other matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import random
import librosa
import queue
import threading
from typing import Optional
import src.utility as util
import argparse


# By default the random number generator uses the current system time.
# Initializing the random number generator with seed for reproducability:
# rerunning the script produces the same results.
random.seed(a=10)

# style plot
style.use('fivethirtyeight')
style.use('dark_background')

class ChappyVisOne(object):
    """
    Single-subplot live plotting visualization:
    - expects a q where each item is (audio_chunk, float_sample)
    - shows one mel-spectrogram with its colorbar
    - uses threading for improved performance
    """

    def __init__(self, args: dict = None, interval: int = 200, q: Optional[queue.Queue] = None):
        if args is None:
            args = {'hoplength': 512/2, 'samplerate': 48000,
                    'mels': 128/2, 'nfft': 2048/4}
 
        self.args = args
        self.melspec = melSpecVis(self.args)
        self.interval = interval

        # external data q (items: (audio, float_sample))
        self.q = q if q is not None else queue.Queue()
        
        # Internal buffer queue for processed data
        self.buffer_q = queue.Queue(maxsize=50)
        
        # Threading control
        self.running = True
        self.lock = threading.Lock()

        # create figure: single spectrogram
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 6))

        # Prepare initial spectrogram and colorbar
        init_audio = np.zeros(self.args.hoplength * 10, dtype=np.float32)
        self.s_db = self.melspec.get_mel_spec(init_audio)
        
        self.spec_im = librosa.display.specshow(self.s_db,
                                                sr=self.args.samplerate,
                                                hop_length=self.args.hoplength,
                                                x_axis='time',
                                                y_axis='mel',
                                                cmap=self.melspec.cmap,
                                                ax=self.ax)
        self.spec_cbar = plt.colorbar(self.spec_im, ax=self.ax, format='%+2.0f dB')
        self.ax.set_title('Mel spectrogram')

        self.ani = None
        plt.tight_layout()
        
        # Start data processing thread
        self.data_thread = threading.Thread(target=self._data_processor, daemon=True)
        self.data_thread.start()
        
        print(f"ChappyVisOne.__init__ completed. Figure: {self.fig}, Axes: {self.ax}")

    def _data_processor(self):
        """
        Background thread that processes incoming audio data and prepares spectrograms.
        """
        last_log_time = 0
    
        while self.running:
            try:
                current_time = threading.Event().wait(0)  # Non-blocking

                # Log status every 5 seconds
                import time
                if time.time() - last_log_time > 5:
                    print(f"[DataProcessor] Input queue: {self.q.qsize()}, Buffer: {self.buffer_q.qsize()}")
                    last_log_time = time.time()

                # Collect data from input queue
                items = []
                while self.q.qsize() > 0 and len(items) < 1000:
                    try:
                        items.append(self.q.get_nowait())
                    except queue.Empty:
                        break
                        
                if items:   
                    # Aggregate audio chunks
                    audio_chunks = []

                    for item in items:
                        audio_chunks.append(item)

                    audio = np.concatenate(audio_chunks)

                    # Compute spectrogram in background thread
                    s_db = self.melspec.get_mel_spec(audio)

                    # Put processed data in buffer queue
                    try:
                        self.buffer_q.put_nowait((audio, s_db))
                    except queue.Full:
                        # Drop oldest if buffer is full
                        try:
                            self.buffer_q.get_nowait()
                            self.buffer_q.put_nowait((audio, s_db))
                        except:
                            pass
                else:
                    # Small sleep to prevent busy-waiting
                    threading.Event().wait(0.01)
                
            except Exception as e:
                print(f"Error in data processor thread: {e}")
                import traceback
                traceback.print_exc()
                threading.Event().wait(0.01)

    def read_data(self):
        """
        Pop ALL items from buffer queue and aggregate them.
        Returns tuple (audio_chunk, s_db) combining all buffered spectrograms.
        """
        items = []
    
        # Drain the entire buffer queue (or a portion if too large)
        if self.buffer_q.qsize() > 5:
            while self.buffer_q.qsize() > 0:
                try:
                    items.append(self.buffer_q.get_nowait())
                except queue.Empty:
                    break
    
        if items:
            # Aggregate all audio chunks and spectrograms
            audio_chunks = []
            s_db_chunks = []
            
            for audio, s_db in items:
                audio_chunks.append(audio)
                s_db_chunks.append(s_db)
            
            # Concatenate audio
            combined_audio = np.concatenate(audio_chunks)
            
            # Concatenate spectrograms along time axis (axis=1)
            combined_s_db = np.concatenate(s_db_chunks, axis=1)
            
            return (combined_audio, combined_s_db)
    
        # fallback synthetic data
        n = int(self.args.hoplength * 10)
        audio = np.zeros(n, dtype=np.float32)
        s_db = self.melspec.get_mel_spec(audio)
        return (audio, s_db)

    def animate(self, frame):
        # obtain pre-processed data
        new_audio, s_db = self.read_data()
        
        with self.lock:
            self.s_db = np.append(self.s_db, s_db, axis=1)

            # Limit to last 10 seconds of data
            max_frames = int(10 * self.args.samplerate / self.args.hoplength)  # ~938 frames for 10s
            if self.s_db.shape[1] > max_frames:
                self.s_db = self.s_db[:, -max_frames:]
            # update spectrogram (data already computed)
            try:
                if self.spec_cbar is not None:
                    self.spec_cbar.remove()
            except Exception:
                pass
            self.ax.clear()
            self.spec_im = librosa.display.specshow(self.s_db,
                                                    sr=self.args.samplerate,
                                                    hop_length=self.args.hoplength,
                                                    x_axis='time',
                                                    y_axis='mel',
                                                    cmap=self.melspec.cmap,
                                                    ax=self.ax)
            self.spec_cbar = plt.colorbar(self.spec_im, ax=self.ax, format='%+2.0f dB')
            self.ax.set_title('Mel spectrogram')

        return (self.spec_im,)

    def update(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            interval=self.interval,
            blit=False
        )
        return self.ani
    
    def stop(self):
        """Clean shutdown of threading"""
        self.running = False
        if self.data_thread.is_alive():
            self.data_thread.join(timeout=1.0)


class ChappyVisFour(object):
    """
    Four-subplot live plotting visualization:
    - Two mel-spectrograms (top row) - last 10 seconds
    - Log-spectral distance line plot (middle) - last 10 seconds
    - Float array values line plot (bottom) - last 10 seconds
    - Expects queue items as: (audio_chunk1, audio_chunk2, float_array)
    """

    def __init__(self, args: dict = None, x_len: int = 100, y_range: list[float] = [-1, 1],
                 interval: int = 200, q: Optional[queue.Queue] = None):
        if args is None:
            args = {'hoplength': 256, 'samplerate': 48000,
                    'mels': 64, 'nfft': 1024}
 
        self.args = args
        self.melspec = melSpecVis(self.args)
        self.x_len = x_len
        self.y_range = list(y_range)
        self.interval = interval

        # External data queue (items: (audio1, audio2, float_array))
        self.q = q if q is not None else queue.Queue()
        
        # Internal buffer queue for processed data
        self.buffer_q = queue.Queue(maxsize=50)
        
        # Threading control
        self.running = True
        self.lock = threading.Lock()

        # Create figure: 2x2 grid
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 10))
        self.spec_ax1 = self.axes[0, 0]
        self.spec_ax2 = self.axes[0, 1]
        self.lsd_ax = self.axes[1, 0]
        self.float_ax = self.axes[1, 1]

        # Calculate max frames for 10 seconds
        self.max_frames = int(10 * self.args.samplerate / self.args.hoplength)

        # Initialize accumulated spectrograms
        init_audio = np.zeros(self.args.hoplength * 10, dtype=np.float32)
        s_db = self.melspec.get_mel_spec(init_audio)
        self.s_db1 = s_db
        self.s_db2 = s_db

        # Prepare line plot buffers for log-spectral distance (as list, like float_ys)
        self.lsd_ys = []
        self.lsd_ax.set_xlim([0, 1000])
        self.lsd_ax.set_ylim([0, 50])  # Typical LSD range
        self.lsd_ax.grid(color='green', linestyle='--', linewidth=0.25)
        self.lsd_ax.set_title('Log-Spectral Distance (Last 10s)')
        self.lsd_ax.set_xlabel('Sample Index')
        self.lsd_ax.set_ylabel('LSD (dB)')
        self.lsd_line, = self.lsd_ax.plot([], [], color='#ff6b6b')

        # Prepare line plot buffers for float array values
        self.float_ys = []
        self.float_ax.set_xlim([0, 1000])  # Will auto-adjust
        self.float_ax.set_ylim(self.y_range)
        self.float_ax.grid(color='green', linestyle='--', linewidth=0.25)
        self.float_ax.set_title('Float Array Values (Last 10s)')
        self.float_ax.set_xlabel('Sample Index')
        self.float_ax.set_ylabel('Value')
        self.float_line, = self.float_ax.plot([], [], color='#33ebff')

        # Prepare initial spectrograms
        self.spec_im1 = librosa.display.specshow(self.s_db1,
                                                 sr=self.args.samplerate,
                                                 hop_length=self.args.hoplength,
                                                 x_axis='time',
                                                 y_axis='mel',
                                                 cmap=self.melspec.cmap,
                                                 ax=self.spec_ax1)
        self.spec_cbar1 = plt.colorbar(self.spec_im1, ax=self.spec_ax1, format='%+2.0f dB')
        self.spec_ax1.set_title('Input Mel Spectrogram (Last 10s)')

        self.spec_im2 = librosa.display.specshow(self.s_db2,
                                                 sr=self.args.samplerate,
                                                 hop_length=self.args.hoplength,
                                                 x_axis='time',
                                                 y_axis='mel',
                                                 cmap=self.melspec.cmap,
                                                 ax=self.spec_ax2)
        self.spec_cbar2 = plt.colorbar(self.spec_im2, ax=self.spec_ax2, format='%+2.0f dB')
        self.spec_ax2.set_title('Output Mel Spectrogram (Last 10s)')

        self.ani = None
        plt.tight_layout()
        
        # Start data processing thread
        self.data_thread = threading.Thread(target=self._data_processor, daemon=True)
        self.data_thread.start()
        
        print(f"ChappyVisFour.__init__ completed.")

    def _compute_lsd(self, s_db1, s_db2):
        """Compute log-spectral distance between two spectrograms"""
        # Ensure same shape
        min_frames = min(s_db1.shape[1], s_db2.shape[1])
        s_db1 = s_db1[:, :min_frames]
        s_db2 = s_db2[:, :min_frames]
        
        # Convert dB back to power for proper distance calculation
        # LSD = sqrt(mean((log(S1) - log(S2))^2))
        diff = (s_db1 - s_db2) ** 2
        lsd = np.sqrt(np.mean(diff))
        return lsd

    def _data_processor(self):
        """Background thread for processing audio data"""
        last_log_time = 0
        
        while self.running:
            try:
                import time
                if time.time() - last_log_time > 5:
                    print(f"[ChappyVisFour] Input queue: {self.q.qsize()}, Buffer: {self.buffer_q.qsize()}")
                    last_log_time = time.time()

                items = []
                while self.q.qsize() > 0 and len(items) < 1000:
                    try:
                        items.append(self.q.get_nowait())
                    except queue.Empty:
                        break
                
                if items:
                    # Aggregate data
                    audio1_chunks = []
                    audio2_chunks = []
                    float_arrays = []
                    
                    for item in items:
                        audio1_chunks.append(item[0])
                        audio2_chunks.append(item[1])
                        # Handle both numpy arrays and single values
                        float_val = item[2]
                        if isinstance(float_val, np.ndarray):
                            float_arrays.extend(float_val.flatten().tolist())
                        else:
                            float_arrays.append(float_val)
                    
                    audio1 = np.concatenate(audio1_chunks)
                    audio2 = np.concatenate(audio2_chunks)
                    
                    # Compute spectrograms
                    s_db1 = self.melspec.get_mel_spec(audio1)
                    s_db2 = self.melspec.get_mel_spec(audio2)
                    
                    # Compute log-spectral distance
                    lsd = self._compute_lsd(s_db1, s_db2)
                    
                    # Put processed data in buffer (pass the full float array)
                    try:
                        self.buffer_q.put_nowait((s_db1, s_db2, lsd, float_arrays))
                    except queue.Full:
                        try:
                            self.buffer_q.get_nowait()
                            self.buffer_q.put_nowait((s_db1, s_db2, lsd, float_arrays))
                        except:
                            pass
                else:
                    threading.Event().wait(0.01)
                    
            except Exception as e:
                print(f"Error in data processor: {e}")
                import traceback
                traceback.print_exc()
                threading.Event().wait(0.01)

    def read_data(self):
        """
        Pop ALL items from buffer queue and aggregate them.
        Returns tuple (s_db1, s_db2, lsd, float_arrays) combining all buffered data.
        """
        items = []
        
        # Drain the entire buffer queue
        if self.buffer_q.qsize() > 5:
            while self.buffer_q.qsize() > 0:
                try:
                    items.append(self.buffer_q.get_nowait())
                except queue.Empty:
                    break
        
        if items:
            # Aggregate all spectrograms and data
            s_db1_chunks = []
            s_db2_chunks = []
            lsd_values = []
            all_float_arrays = []
            
            for s_db1, s_db2, lsd, float_arrays in items:
                s_db1_chunks.append(s_db1)
                s_db2_chunks.append(s_db2)
                lsd_values.append(lsd)
                all_float_arrays.extend(float_arrays)
            
            # Concatenate spectrograms along time axis (axis=1)
            combined_s_db1 = np.concatenate(s_db1_chunks, axis=1)
            combined_s_db2 = np.concatenate(s_db2_chunks, axis=1)
            
            return (combined_s_db1, combined_s_db2, lsd_values, all_float_arrays)
        
        # Fallback synthetic data
        n = int(self.args.hoplength * 10)
        audio = np.zeros(n, dtype=np.float32)
        s_db = self.melspec.get_mel_spec(audio)
        return (s_db, s_db, [0.0], [0.0])

    def animate(self, frame):
        """Animation function"""
        s_db1, s_db2, lsd_values, float_arrays = self.read_data()
        
        with self.lock:
            # Accumulate spectrograms (last 10 seconds)
            self.s_db1 = np.append(self.s_db1, s_db1, axis=1)
            self.s_db2 = np.append(self.s_db2, s_db2, axis=1)
            
            # Extend LSD values list (same as float_arrays)
            self.lsd_ys.extend(lsd_values)
            self.float_ys.extend(float_arrays)

            # Estimate max samples for 10 seconds
            max_float_samples = self.max_frames * 100  # Adjust multiplier as needed
            
            # Limit to last 10 seconds
            if self.s_db1.shape[1] > self.max_frames:
                self.s_db1 = self.s_db1[:, -self.max_frames:]
            if self.s_db2.shape[1] > self.max_frames:
                self.s_db2 = self.s_db2[:, -self.max_frames:]

            # Update LSD line plot (same as float_line)
            if len(self.lsd_ys) > max_float_samples:
                self.lsd_ys = self.lsd_ys[-max_float_samples:]

            lsd_xs = list(range(len(self.lsd_ys)))
            self.lsd_line.set_xdata(lsd_xs)
            self.lsd_line.set_ydata(self.lsd_ys)
            
            # Update x-axis limits (same as float plot)
            if len(lsd_xs) > 0:
                self.lsd_ax.set_xlim([0, max(lsd_xs)])
            
            # Auto-scale LSD y-axis if needed
            if len(self.lsd_ys) > 0:
                min_lsd = min(self.lsd_ys)
                max_lsd = max(self.lsd_ys)
                current_ylim = self.lsd_ax.get_ylim()
                
                if min_lsd < current_ylim[0] or max_lsd > current_ylim[1]:
                    margin = (max_lsd - min_lsd) * 0.1 if max_lsd != min_lsd else 0.1
                    self.lsd_ax.set_ylim([min_lsd - margin, max_lsd + margin])
            
            # Update float array line plot
            if len(self.float_ys) > max_float_samples:
                self.float_ys = self.float_ys[-max_float_samples:]
            
            float_xs = list(range(len(self.float_ys)))
            self.float_line.set_xdata(float_xs)
            self.float_line.set_ydata(self.float_ys)
            
            # Update x-axis limits
            if len(float_xs) > 0:
                self.float_ax.set_xlim([0, max(float_xs)])
            
            # Auto-scale float y-axis if needed
            if len(self.float_ys) > 0:
                min_val = min(self.float_ys)
                max_val = max(self.float_ys)
                current_ylim = self.float_ax.get_ylim()
                
                if min_val < current_ylim[0] or max_val > current_ylim[1]:
                    margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
                    self.float_ax.set_ylim([min_val - margin, max_val + margin])

            # Update spectrogram 1
            try:
                if self.spec_cbar1 is not None:
                    self.spec_cbar1.remove()
            except Exception:
                pass
            self.spec_ax1.clear()
            self.spec_im1 = librosa.display.specshow(self.s_db1,
                                                     sr=self.args.samplerate,
                                                     hop_length=self.args.hoplength,
                                                     x_axis='time',
                                                     y_axis='mel',
                                                     cmap=self.melspec.cmap,
                                                     ax=self.spec_ax1)
            self.spec_cbar1 = plt.colorbar(self.spec_im1, ax=self.spec_ax1, format='%+2.0f dB')
            self.spec_ax1.set_title('Input Mel Spectrogram (Last 10s)')

            # Update spectrogram 2
            try:
                if self.spec_cbar2 is not None:
                    self.spec_cbar2.remove()
            except Exception:
                pass
            self.spec_ax2.clear()
            self.spec_im2 = librosa.display.specshow(self.s_db2,
                                                     sr=self.args.samplerate,
                                                     hop_length=self.args.hoplength,
                                                     x_axis='time',
                                                     y_axis='mel',
                                                     cmap=self.melspec.cmap,
                                                     ax=self.spec_ax2)
            self.spec_cbar2 = plt.colorbar(self.spec_im2, ax=self.spec_ax2, format='%+2.0f dB')
            self.spec_ax2.set_title('Output Mel Spectrogram (Last 10s)')

        return (self.spec_im1, self.spec_im2, self.lsd_line, self.float_line)

    def update(self):
        """Start animation"""
        self.ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            interval=self.interval,
            blit=False
        )
        return self.ani
    
    def stop(self):
        """Clean shutdown"""
        self.running = False
        if self.data_thread.is_alive():
            self.data_thread.join(timeout=1.0)


class melSpecVis():
    def __init__(self, args, cmap='inferno'):
        self.args = args
        self.cmap = cmap
    

    def get_mel_spec(self, audio):
        s = librosa.feature.melspectrogram(y=audio,
                                                sr=self.args.samplerate,
                                                n_mels=self.args.mels,
                                                n_fft=self.args.nfft,
                                                hop_length=self.args.hoplength)
        s_db = librosa.power_to_db(s, ref=np.max)
        """
        times = librosa.times_like(s_db,
                                    sr=self.args.samplerate,
                                    hop_length=self.args.hoplength)
        mel_freqs = librosa.mel_frequencies(n_mels=self.args.mels,
                                            fmin=0,
                                            fmax=self.args.samplerate/2)
        return s_db, times, mel_freqs
        """
        return s_db

    def mel_spec_img(self, s_db, ax):
        img = librosa.display.specshow(s_db, sr=self.args.samplerate, hop_length=self.args.hoplength,
                                    x_axis='time', y_axis='mel', cmap=self.cmap, ax=ax)
        cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel spectrogram')
        # return fig, ax, img

def main():
    """Launcher"""
    filename = os.path.basename(__file__)
    print(f'Start of {filename} \n')
    parser = argparse.ArgumentParser(add_help=False)
    args, mapping, q = util.read_input_arguments(parser)
    args.samplerate = 48000
    args.hoplength = 512
    args.nfft = 2048
    args.mels = 1024
    chappy = ChappyVisThree(args=args)
    chappy.update()
    # Set up plot to call animate() function periodically
    try:
        plt.show()
        print('Closed Plot')
        print('End of', filename)
    except KeyboardInterrupt:
        print('Keyboard interrupt occurred')
        print('End of', filename)


if __name__ == "__main__":
    main()