import sys, os, pathlib, importlib.util
# add project root to sys.path so 'models' package (models/__init__.py) is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import argparse
import sounddevice as sd
import utility
import callback_func
import time
import queue
import src.visualization as vis
parser = argparse.ArgumentParser(add_help=False)
args, mapping, q = utility.read_input_arguments(parser)
plt.ioff()  # Turn OFF interactive mode so plt.show() blocks 
"""
vis_in_obj = utility.alt_StreamVisualization(args.chunk, args.samplerate,
                                                n_mels=128,
                                                n_fft=1024,
                                                hop_length=256)
"""
#vis_out_obj = utility.alt_StreamVisualization(args.chunk, args.samplerate,
#                                                n_mels=128,
#                                                n_fft=1024,
#                                                hop_length=256)

# load model
model = callback_func.audio_model_passthrough(model_type='FLowHigh', arguments=args)

stream = sd.Stream(
        device=args.device,
        channels=1,  #  max(args.channels),
        samplerate=args.samplerate,
        blocksize=args.chunk,
        callback=model.stream_callback(arguments=args, mapping=mapping, queue=q)
        )

with stream:
        filename = os.path.basename(__file__)
        print(f'Start of {filename} \n')
        watcher = vis.ChappyVisThree(args=args, x_len=2048, q=q, interval=500)
        print(f"Figure created: {watcher.fig}")
        print(f"Calling watcher.update()...")
        anim = watcher.update()
        print(f"Animation object: {anim}")
        print(f"About to call plt.show()...")
        print(f"Interactive mode: {plt.isinteractive()}")
        try:
                plt.show(block=True)  # Explicitly request blocking
                print('Closed Plot')
        except:
                quit()