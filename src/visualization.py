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
# from collections import deque
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

class ChappyVisThree(object):
    """
    Three-subplot live plotting visualization adapted to:
    - expect a q where each item is (audio_chunk1, audio_chunk2, float_sample)
    - show two mel-spectrograms (axes[0], axes[1]) each with its colorbar
    - show one live line plot (axes[2]) updated with the float samples
    """

    def __init__(self, args: dict = None, x_len: int = 10, y_range: list[float] = [-1, 1],
                 interval: int = 200, q: Optional[queue.Queue] = None):
        if args is None:
            args = {'hoplength': 512, 'samplerate': 48000,
                    'mels': 64, 'nfft': 2048}
 
        self.args = args
        self.melspec = self.start_melspec()
        self.x_len = x_len
        self.y_range = list(y_range)
        self.interval = interval

        # external data q (items: (audio1, audio2, float_sample))
        self.q = q if q is not None else queue.Queue()

        # create figure: two spectrograms on top, line plot at bottom
        self.fig, self.axes = plt.subplots(3, 1, figsize=(14, 9))
        self.spec_axes = self.axes[:2]
        self.line_ax = self.axes[2]

        # Prepare line plot buffers
        self.xs = list(range(0, self.x_len))
        self.ys = [0] * self.x_len
        self.line_ax.set_xlim([0, self.x_len - 1])
        self.line_ax.set_ylim(self.y_range)
        self.line_ax.grid(color='green', linestyle='--', linewidth=0.5)
        self.line_ax.axes.xaxis.set_ticklabels([])
        self.line_ax.set_title('Live Value')
        self.line_ax.set_xlabel('Time')
        self.line_ax.set_ylabel('Value')
        self.line, = self.line_ax.plot(self.xs, self.ys, color='#33ebff')

        # Prepare initial spectrograms and colorbars
        init_audio = np.zeros(self.args.hoplength * self.x_len, dtype=np.float32)
        self.audio1 = init_audio
        self.audio2 = init_audio
        s_db1, times1, mel_freqs1 = self.melspec.get_mel_spec(self.audio1)
        s_db2, times2, mel_freqs2 = self.melspec.get_mel_spec(self.audio2)
        self.spec_im1 = librosa.display.specshow(s_db1,
                                                 sr=self.args.samplerate,
                                                 hop_length=self.args.hoplength,
                                                 x_axis='time',
                                                 y_axis='mel',
                                                 cmap=self.melspec.cmap,
                                                 ax=self.spec_axes[0])
        self.spec_cbar1 = plt.colorbar(self.spec_im1, ax=self.spec_axes[0], format='%+2.0f dB')
        self.spec_axes[0].set_title('Mel spectrogram 1')

        self.spec_im2 = librosa.display.specshow(s_db2,
                                                 sr=self.args.samplerate,
                                                 hop_length=self.args.hoplength,
                                                 x_axis='time',
                                                 y_axis='mel',
                                                 cmap=self.melspec.cmap,
                                                 ax=self.spec_axes[1])
        self.spec_cbar2 = plt.colorbar(self.spec_im2, ax=self.spec_axes[1], format='%+2.0f dB')
        self.spec_axes[1].set_title('Mel spectrogram 2')

        self.ani = None
        plt.tight_layout()
        print(f"ChappyVisThree.__init__ completed. Figure: {self.fig}, Axes: {self.axes}")


    def start_melspec(self):
        return melSpecVis(self.args)

    def read_data(self):
        """
        Pop one item from the q if available, otherwise generate synthetic:
        returns tuple (audio_chunk1, audio_chunk2, float_sample)
        """
        
        if self.q.qsize() > 0:
            try:
                item = [[],[],[]]
                for _ in range(self.q.qsize()):
                    tmp_item = self.q.get()
                    item[0].append(tmp_item[0][:])
                    item[1].append(tmp_item[1][:])
                    item[2].append(tmp_item[2][:])
                    # Expect item to be (audio1, audio2, float_val)
                return item
            except Exception:
                pass
        
        # fallback synthetic data
        n = int(self.args.hoplength * self.x_len)
        audio1 = np.random.randn(n).astype(np.float32)
        audio2 = np.random.randn(n).astype(np.float32)
        float_sample = random.random() - 0.5
        return (audio1, audio2, float_sample)

    def animate(self, frame):
        # obtain data: two audio chunks and one float

        new_audio1, new_audio2, float_sample = self.read_data()
        self.audio1 = np.append(self.audio1, new_audio1)
        self.audio2 = np.append(self.audio2, new_audio2)

        # update line plot with the float sample
        self.ys.append(float_sample)
        self.ys = self.ys[-self.x_len:]
        ylimits = self.line_ax.get_ylim()
        rescale = False
        if float_sample < ylimits[0]:
            new_limit = abs(float_sample - 0.1)
            self.line_ax.set_ylim(-new_limit, new_limit)
            rescale = True
        if float_sample > ylimits[1]:
            new_limit = float_sample + 0.1
            self.line_ax.set_ylim(-new_limit, new_limit)
            rescale = True
        self.line.set_ydata(self.ys)

        # update spectrogram 1
        try:
            if self.spec_cbar1 is not None:
                self.spec_cbar1.remove()
        except Exception:
            pass
        self.spec_axes[0].clear()
        s_db1, _, _ = self.melspec.get_mel_spec(self.audio1)
        self.spec_im1 = librosa.display.specshow(s_db1,
                                                 sr=self.args.samplerate,
                                                 hop_length=self.args.hoplength,
                                                 x_axis='time',
                                                 y_axis='mel',
                                                 cmap=self.melspec.cmap,
                                                 ax=self.spec_axes[0])
        self.spec_cbar1 = plt.colorbar(self.spec_im1, ax=self.spec_axes[0], format='%+2.0f dB')
        self.spec_axes[0].set_title('Mel spectrogram 1')

        # update spectrogram 2
        try:
            if self.spec_cbar2 is not None:
                self.spec_cbar2.remove()
        except Exception:
            pass
        self.spec_axes[1].clear()
        s_db2, _, _ = self.melspec.get_mel_spec(self.audio2)
        self.spec_im2 = librosa.display.specshow(s_db2,
                                                 sr=self.args.samplerate,
                                                 hop_length=self.args.hoplength,
                                                 x_axis='time',
                                                 y_axis='mel',
                                                 cmap=self.melspec.cmap,
                                                 ax=self.spec_axes[1])
        self.spec_cbar2 = plt.colorbar(self.spec_im2, ax=self.spec_axes[1], format='%+2.0f dB')
        self.spec_axes[1].set_title('Mel spectrogram 2')

        if rescale:
            self.fig.canvas.draw()

        # return updated artists
        return (self.spec_im1, self.spec_im2, self.line)

    def update(self):
        self.ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            interval=self.interval,
            blit=False
        )
        return self.ani        # ...existing code in __init__...


class melSpecVis():
    def __init__(self, args, cmap='inferno'):
        self.args = args
        self.cmap = cmap
    

    def get_mel_spec(self, audio):
        s = librosa.feature.melspectrogram(y=audio,
                                                sr=self.args.samplerate,
                                                n_mels=self.args.mels,
                                                n_fft=self.args.nfft)
        s_db = librosa.power_to_db(s, ref=np.max)
        times = librosa.times_like(s_db,
                                    sr=self.args.samplerate,
                                    hop_length=self.args.hoplength)
        mel_freqs = librosa.mel_frequencies(n_mels=self.args.mels,
                                            fmin=0,
                                            fmax=self.args.samplerate/2)
        return s_db, times, mel_freqs

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