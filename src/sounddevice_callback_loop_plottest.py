import sys, os, pathlib, importlib.util
# add project root to sys.path so 'models' package (models/__init__.py) is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from scipy.fftpack import fft

import argparse
import numpy as np
import sounddevice as sd
import utility
import callback_func
parser = argparse.ArgumentParser(add_help=False)
args, mapping, q = utility.read_input_arguments(parser)

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

"""
stream = sd.InputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate, callback=model.audio_in_callback(arguments=args, mapping=mapping, queue=q))


out_stream = sd.OutputStream(
        device=args.device, channels=max(args.channels),
        samplerate=args.samplerate)  # , callback=callback_func.audio_callback(arguments=args, mapping=mapping, queue=q))

"""

stream = sd.Stream(
        device=args.device,
        channels=1,  #  max(args.channels),
        samplerate=args.samplerate,
        blocksize=args.chunk,
        callback=model.stream_callback(arguments=args, mapping=mapping, queue=q)
        )

with stream:
        #while True:
        data_in = q[0].get()
        data_out = q[1].get()
        
        # vis_out_obj.update_trace(data_out.squeeze())
              
    
 
    