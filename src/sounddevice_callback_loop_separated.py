import sys, os, pathlib, importlib.util
# add project root to sys.path so 'models' package (models/__init__.py) is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import argparse
import sounddevice as sd
import utility
import callback_func
import time
import threading
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
model = callback_func.audio_model_pair(model_type='FLowHigh', arguments=args)

original_callback = model.in_stream_callback(arguments=args, mapping=mapping, queue=q[0])

callback_error = [None]  # Store error from callback
callback_count = [0]  # Count callbacks

def safe_callback(indata, frames, time_info, status):
    try:
        callback_count[0] += 1
        if status:
            print(f"Callback status: {status}")
        result = original_callback(indata, frames, time_info, status)
        
        # Check if queue is being written to
        if callback_count[0] % 100 == 0:
            print(f"Callback #{callback_count[0]}, in-queue size: {q[0].qsize()}")
        
        return result
    except Exception as e:
        callback_error[0] = e
        print(f"ERROR in callback: {e}")
        import traceback
        traceback.print_exc()
        # Continue to prevent stream from dying

stream = sd.InputStream(
        device=21,
        channels=1,
        samplerate=args.samplerate,
        blocksize=args.chunk,
        callback=safe_callback
        )

with stream:
      
        def monitor_queue():
                while True:
                        time.sleep(5)
                        size = q[0].qsize()
                        buffer_size = watcher.buffer_q.qsize()
                        print(f"Queue size: {size}, Buffer: {buffer_size}, Callbacks: {callback_count[0]}")

                        if callback_error[0]:
                                print(f"CALLBACK ERROR DETECTED: {callback_error[0]}")

                        if size == 0:
                                print("WARNING: Input queue is empty!")
                        if size >= 90:
                                print(f"WARNING: Input queue near full! ({size}/100)")
        
        monitor_thread = threading.Thread(target=monitor_queue, daemon=True)
        monitor_thread.start()
    
        watcher = vis.ChappyVisOne(args=args, q=q[0], interval=200)
        anim = watcher.update()
        
        try:
                plt.show(block=True)
                print('Closed Plot')
        except KeyboardInterrupt:
                print("Interrupted by user")
        finally:
                watcher.stop()
                print(f"Final callback count: {callback_count[0]}")
                if callback_error[0]:
                        print(f"Callback error was: {callback_error[0]}")

