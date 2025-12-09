import argparse
import sounddevice as sd
import utility
import callback_func
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

parser = argparse.ArgumentParser(add_help=False)
args, mapping, q = utility.read_input_arguments(parser)

vis_obj = utility.StreamVisualization(args=args)


stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=callback_func.audio_callback(arguments=args, mapping=mapping, queue=q))

"""
out_stream = sd.OutputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate)  # , callback=callback_func.audio_callback(arguments=args, mapping=mapping, queue=q))
"""

ani = FuncAnimation(vis_obj.fig, vis_obj.update_plot(q=q), interval=args.interval, blit=True)

with stream:
    plt.show()

    # while True:
    #for _ in range(10):
        # queue_output.append(q.get())

# plt.plot(queue_output[0])
# plt.show()
