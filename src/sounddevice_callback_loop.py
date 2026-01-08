import sys, os, pathlib, importlib.util
# add project root to sys.path so 'models' package (models/__init__.py) is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import sounddevice as sd
import utility
import callback_func
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

parser = argparse.ArgumentParser(add_help=False)
args, mapping, q = utility.read_input_arguments(parser)

# vis_obj = utility.StreamVisualization(args=args)

# load model
model = callback_func.audio_model_passthrough(model_type='FLowHigh', arguments=args)
# model = callback_func.audio_model_passthrough(model_type='FlashSR', arguments=args)


stream = sd.Stream(
        # device=args.device,
        channels=1,  #  max(args.channels),
        samplerate=args.samplerate,
        blocksize=args.chunk,
        latency='high',
        callback=model.stream_callback(arguments=args, mapping=mapping, queue=q)
        )


with stream:
        while True:
                pass
                # data_in = q.get()
                # print(f'Queue size: {q.qsize()}')
# with stream:
#     while True:
#         ani = FuncAnimation(vis_obj.fig, vis_obj.update_plot(q=q[0]), interval=args.interval, blit=True)
        
        # for _ in range(10):
        # queue_output.append(q.get())

# plt.plot(queue_output[0])
# plt.show()
