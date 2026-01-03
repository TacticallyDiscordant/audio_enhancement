import sys
import time as timing
import numpy as np
from models.FLowHigh_inference import rAI_FLowHigh
from models.VAudioSR_inference import Predictor

def audio_callback(arguments, mapping, queue):
    """
    fancy closure
    """
    def basic_callback(indata, frames, time, status, arguments=arguments, mapping=mapping, queue=queue):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        queue.put(indata[::arguments.downsample, mapping])

    return basic_callback


def audio_out_callback(arguments, mapping, queue):
    """
    fancy closure
    """
    def basic_callback(frames, time, status, arguments=arguments, mapping=mapping, queue=queue):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)

    return basic_callback


class audio_model_passthrough(object):

    def __init__(self, model_type, arguments):

        self.chunck_measure = 2048 // arguments.downsample
        self.process_crap = False

        if model_type == 'FLowHigh':
            self.model = rAI_FLowHigh(
                                        input_sr=48000,
                                        target_sr=48000,
                                        live_mode=True
                                        )
        elif model_type == 'VersatileAudioSR':
            self.model = Predictor()
            self.model.setup(model_name="speech",
                                device='cpu')

    
    def audio_in_callback(self, arguments, mapping, queue):
        """
        fancy closure
        """
        def basic_callback(indata, frames, time, status, arguments=arguments, mapping=mapping, queue=queue):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            # Fancy indexing with mapping creates a (necessary!) copy:
            """
            try:
                modified = self.model.infer(indata.squeeze())
                print("Yes")
            except:
                modified = indata
                print("No")
            """
            print(f"length of indata: {indata.shape}")
            modified = self.model.infer(indata.squeeze())
            indata[0] = modified[0]
            queue.put(indata[::arguments.downsample, mapping])

        return basic_callback


    def stream_callback(self, arguments, mapping, queue):
        """
        fancy closure
        """
        def basic_callback(indata, outdata, frames, time, status, arguments=arguments, mapping=mapping, queue=queue):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            # Fancy indexing with mapping creates a (necessary!) copy:
            into_queue = []
            into_queue.append(indata[::arguments.downsample, mapping].squeeze())
            start_time = timing.perf_counter()
            modified = np.reshape(self.model.infer(indata.T), indata.shape)
            end_time = timing.perf_counter()
            into_queue.append(modified[::arguments.downsample, mapping].squeeze())
            into_queue.append(end_time - start_time)
            outdata[:] = modified
            queue.put(into_queue)

        return basic_callback
