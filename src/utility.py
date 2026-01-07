import argparse
import torch
import queue
import sounddevice as sd
import numpy as np
from scipy import signal
import librosa
import matplotlib.pyplot as plt
import time
from functools import wraps

def timer(func):
    """A decorator that prints the time a function takes to execute."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f}s")
        return result
    return wrapper

@timer
def example_function(delay):
    """A simple function that pauses for a given time."""
    time.sleep(delay)

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
        '-b', '--blocksize', type=int, default=1024, help='block size (in samples)')
    parser.add_argument(
        '-r', '--samplerate', type=int, default=48000, help='sampling rate of audio device')
    parser.add_argument(
        '-n', '--downsample', type=int, default=1, metavar='N',
        help='display every Nth sample (default: %(default)s)')
    parser.add_argument(
        '-c', '--chunk', type=int, default=2048, metavar='CHUNK',
        help='length of stream required for processing (default: %(default)s)')
    parser.add_argument(
        '-m', '--mels', type=int, default=1024, metavar='NMELS',
        help='Number of Mel bands (default: %(default)s)')
    parser.add_argument(
        '-f', '--nfft', type=int, default=2048, metavar='NFFT',
        help='Length of the signal for visualisation FFT (default: %(default)s)')
    parser.add_argument(
        '-o', '--hoplength', type=int, default=512, metavar='HOPLENGTH',
        help='Hop length for FFT (default: %(default)s)')
    args = parser.parse_args(remaining)
    if any(c < 1 for c in args.channels):
        parser.error('argument CHANNEL: must be >= 1')
    mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
    # q = queue.Queue()
    q = queue.Queue(maxsize=100)
    return args, mapping, q

class LSD(object):
    EPS = 1e-12

    def __init__(self, estimation, target, args):
        self.args = args
        self.estimation = self.to_spectrogram(estimation)
        self.target = self.to_spectrogram(target)
        self.lsd = self.log_spectral_distance()


    def log_spectral_distance(self):
        lsd = torch.log10(self.target**2 / ((self.estimation + self.EPS) ** 2) + self.EPS) ** 2
        lsd = torch.mean(torch.mean(lsd, axis=3) ** 0.5, axis=2)
        return lsd[..., None, None]

    def to_spectrogram(self, audio) :
        f = np.abs(librosa.stft(audio, hop_length=self.args.hop_length, n_fft=self.args.n_fft))
        f = np.transpose(f, (1, 0))
        f = torch.tensor(f[None, None, ...])
        return f
