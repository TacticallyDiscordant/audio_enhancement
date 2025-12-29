import argparse
import queue
import sounddevice as sd
import numpy as np
from scipy import signal
import librosa
import matplotlib.pyplot as plt
# plotly offline
import plotly.offline as pyo
# from plotly.offline import init_notebook_mode #to plot in jupyter notebook
import plotly.graph_objs as go

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def read_input_arguments(parser):
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
        help='input channels to plot (default: the first)')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-w', '--window', type=float, default=200, metavar='DURATION',
        help='visible time slot (default: %(default)s ms)')
    parser.add_argument(
        '-i', '--interval', type=float, default=30,
        help='minimum time between plot updates (default: %(default)s ms)')
    parser.add_argument(
        '-b', '--blocksize', type=int, help='block size (in samples)')
    parser.add_argument(
        '-r', '--samplerate', default=48000, type=float, help='sampling rate of audio device')
    parser.add_argument(
        '-n', '--downsample', type=int, default=1, metavar='N',
        help='display every Nth sample (default: %(default)s)')
    parser.add_argument(
        '-c', '--chunk', type=int, default=2048, metavar='CHUNK',
        help='length of stream required for processing (default: %(default)s)')
    parser.add_argument(
        '--mels', type=int, default=1024, metavar='NMELS',
        help='Number of Mel bands (default: %(default)s)'
                        ),
    parser.add_argument(
        '--nfft', type=int, default=2048, metavar='NFFT',
        help='Length of the signal for visualisation FFT (default: %(default)s)'
                        ),
    parser.add_argument(
        '--hoplength', type=int, default=512, metavar='HOPLENGTH',
        help='Hop length for FFT (default: %(default)s)'
                        )
    args = parser.parse_args(remaining)
    if any(c < 1 for c in args.channels):
        parser.error('argument CHANNEL: must be >= 1')
    mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
    q = [queue.Queue(), queue.Queue()]
    return args, mapping, q



class StreamVisualization():
    def __init__(self, args) -> None:

        self.args = args
        if self.args.samplerate is None:
            device_info = sd.query_devices(self.args.device, 'input')
            self.args.samplerate = device_info['default_samplerate']

        length = int(self.args.window * self.args.samplerate / (1000 * self.args.downsample))
    
        self.plotdata = np.zeros((length, len(args.channels)))

        self.fig, self.ax = plt.subplots()
        self.lines = self.ax.plot(self.plotdata)
        self.set_appearance()
        
    def set_appearance(self) -> None:
        if len(self.args.channels) > 1:
            self.ax.legend([f'channel {c}' for c in self.args.channels],
                      loc='lower left', ncol=len(self.args.channels))
        self.ax.axis((0, len(self.plotdata), -1, 1))
        self.ax.set_yticks([0])
        self.ax.yaxis.grid(True)
        self.ax.tick_params(bottom=False, top=False, labelbottom=False,
                    right=False, left=False, labelleft=False)
        self.fig.tight_layout(pad=0)



    def update_plot(self, q):
        """This is called by matplotlib for each plot update.

        Typically, audio callbacks happen more frequently than plot updates,
        therefore the queue tends to contain multiple blocks of audio data.

        """
        while True:
            try:
                data = q.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            self.plotdata = np.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:, :] = data
        lines = self.lines
        for column, line in enumerate(lines):
            line.set_ydata(self.plotdata[:, column])
        self.lines = lines
        return self.lines
        

class alt_StreamVisualization():
    def __init__(self, chunk_size, sampling_rate=48000,
                        n_mels = 1024,
                        n_fft = 2048,
                        hop_length = 512):
        #Spectrogram
        self.sr = sampling_rate
        self.chunk = chunk_size
        init_audio = librosa.tone(880, sr=self.sr, length=self.chunk)
        update_audio = librosa.tone(2000, sr=self.sr, length=self.chunk)
        # plt.figure()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        # freqs, bins, Pxx = signal.spectrogram(Audiodata, fs,window = w,nfft=N)
        init_trace = self.make_trace(init_audio)
        self.fig = self.figure_init(init_trace)

    def get_mel_spec(self, audio):
        d = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft)
        d_db = librosa.power_to_db(d, ref=np.max)
        times = librosa.times_like(d_db, sr=self.sr, hop_length=self.hop_length)
        mel_freqs = librosa.mel_frequencies(n_mels=self.n_mels, fmin=0, fmax=self.sr/2)
        return d_db, times, mel_freqs

    def make_trace(self, audio):
        d_db, times, mel_freqs = self.get_mel_spec(audio)
        trace = go.Heatmap(z=d_db,
                            x=times,
                            y=mel_freqs,
                            colorscale='Inferno', # Use a visually appealing color scale
                            colorbar={'title': 'Power (dB)'})
        return trace

    def figure_init(self, trace):
        fig = go.Figure(data=[trace])

        # Update layout for better presentation
        fig.update_layout(
                        title='Input Stream',
                        xaxis_title='Time (s)',
                        yaxis_title='Frequency (Mel)',
                        yaxis=dict(type='log') # Often helpful to display mel scale on a log-like axis
                        )
        return fig

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

    def make_figure(self, s_db):
        fig, ax = plt.subplots()

        img = librosa.display.specshow(s_db, sr=self.args.samplerate, hop_length=self.args.hoplength,
                                    x_axis='time', y_axis='mel', cmap=self.cmap, ax=ax)
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel spectrogram')
        return fig, ax, img