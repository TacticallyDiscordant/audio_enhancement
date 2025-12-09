import argparse
import queue
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

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
        '-r', '--samplerate', type=float, help='sampling rate of audio device')
    parser.add_argument(
        '-n', '--downsample', type=int, default=10, metavar='N',
        help='display every Nth sample (default: %(default)s)')
    args = parser.parse_args(remaining)
    if any(c < 1 for c in args.channels):
        parser.error('argument CHANNEL: must be >= 1')
    mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
    q = queue.Queue()
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
        

