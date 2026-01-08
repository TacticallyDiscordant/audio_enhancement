import sys, os, pathlib, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sounddevice as sd
import numpy as np

# Print device 21 info
print("Device 21 info:")
print(sd.query_devices(21))
print("\n")