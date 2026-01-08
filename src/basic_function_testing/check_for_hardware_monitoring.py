import sys, os, pathlib, importlib.util
# add project root to sys.path so 'models' package (models/__init__.py) is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sounddevice as sd
import numpy as np

def silent_callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = 0  # Explicitly silence output
    print(f"Input level: {np.abs(indata).max():.4f}")

with sd.Stream(device=(21, 21), channels=1, callback=silent_callback):
    sd.sleep(10000)