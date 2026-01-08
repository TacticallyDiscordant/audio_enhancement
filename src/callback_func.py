import sys
import time as timing
import numpy as np
import queue as queueing
import threading
from models.FLowHigh_inference import rAI_FLowHigh
from models.VAudioSR_inference import Predictor
from models.FlashSR_inference import FlashSR

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

        elif model_type == 'FlashSR':
                input_sr = 44100
                output_sr = 48000
                self.model = FlashSR(#model_path='./models/weights_and_configs/FlashSR/model.onnx',
                                model_path='./models/weights_and_configs/FlashSR/upsampler.pth',
                                input_sr=input_sr,
                                target_sr=output_sr)

    
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
            modified = np.reshape(self.model.infer(indata.T), indata.shape)
            indata = None

            queue.put(modified[::arguments.downsample, mapping])
            if queue.qsize() < 200:  # load buffer
                outdata = np.zeros(indata.shape)
            else:
                outdata = queue.get_nowait()
            
        return basic_callback

class audio_model_passthrough_with_threading(object):

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
            self.model.setup(model_name="speech", device='cpu')
        
        # Add processing queue and thread
        self.process_queue = queueing.Queue(maxsize=5)
        self.output_queue = queueing.Queue(maxsize=5)
        self.running = True
        self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
        self.process_thread.start()
    
    def _process_worker(self):
        """Background thread for model inference"""
        while self.running:
            try:
                if not self.process_queue.empty():
                    indata = self.process_queue.get(timeout=0.1)
                    try:
                        modified = self.model.infer(indata.T)
                        modified = np.reshape(modified, indata.shape)
                        # Store processed audio
                        try:
                            self.output_queue.put_nowait(modified)
                        except queueing.Full:
                            self.output_queue.get()  # Drop oldest
                            self.output_queue.put_nowait(modified)
                    except Exception as e:
                        print(f"Model inference error: {e}")
                else:
                    threading.Event().wait(0.001)
            except queueing.Empty:
                pass
            except Exception as e:
                print(f"Processing thread error: {e}")
    
    def stream_callback(self, arguments, mapping, queue):
        """
        Non-blocking callback - sends to processing thread
        """
        
        def basic_callback(indata, outdata, frames, time, status, arguments=arguments, mapping=mapping, queue=queue):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            
            try:
                # Send to visualization queue (non-blocking)
                audio_chunk = indata[::arguments.downsample, mapping].squeeze()
                try:
                    queue[0].put_nowait(audio_chunk)
                except queueing.Full:
                    pass
                
                # Send to processing thread (non-blocking)
                try:
                    self.process_queue.put_nowait(indata.copy())
                except queueing.Full:
                    pass  # Skip processing if busy
                
                # Try to get processed audio, otherwise passthrough
                
                try:
                    modified_out = self.output_queue.get_nowait()
                except queueing.Empty:
                    pass
                    # modified_out = indata   # Passthrough if no processed audio ready
                try:
                    queue[1].put_nowait(modified_out)
                except queueing.Full:
                    pass
                
                # outdata[:] = modified_out
                outdata[:] = indata
                    
            except Exception as e:
                print(f"ERROR in stream_callback: {e}")
                outdata[:] = indata
        
        return basic_callback
    
    def stop(self):
        """Clean shutdown"""
        self.running = False
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)


class audio_model_indata_with_threading(object):

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
            self.model.setup(model_name="speech", device='cpu')
        
        # Add processing queue and thread
        self.process_queue = queueing.Queue(maxsize=5)
        self.input_queue = queueing.Queue(maxsize=5)
        self.output_queue = queueing.Queue(maxsize=5)
        self.timing_queue = queueing.Queue(maxsize=5)
        self.running = True
        self.process_thread = threading.Thread(target=self._process_worker, daemon=True)
        self.process_thread.start()
    
    def _process_worker(self):
        """Background thread for model inference"""
        while self.running:
            try:
                if not self.process_queue.empty():
                    indata = self.process_queue.get(timeout=0.1)
                    try:
                        modified, process_time = self.model.timed_infer(indata.T)
                                                # Store processed audio
                        try:
                            self.input_queue.put_nowait(indata)
                            self.output_queue.put_nowait(modified)
                            self.timing_queue.put_nowait(process_time)
                        except queueing.Full:
                            self.input_queue.get()
                            self.output_queue.get()  # Drop oldest
                            self.timing_queue.get()
                            self.input_queue.put_nowait(indata)
                            self.output_queue.put_nowait(modified)
                            self.timing_queue.put_nowait(process_time)
                    except Exception as e:
                        print(f"Model inference error: {e}")
                else:
                    threading.Event().wait(0.001)
            except queueing.Empty:
                pass
            except Exception as e:
                print(f"Processing thread error: {e}")
    
    def stream_indata_callback(self, arguments, mapping, queue):
        """
        Non-blocking callback - sends to processing thread
        """
        
        def basic_callback(indata, frames, time, status, arguments=arguments, mapping=mapping, queue=queue):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            
            queue_list = []
            try:
                # Send to visualization queue (non-blocking)
                audio_chunk = indata[::arguments.downsample, mapping].squeeze()

                
                # Send to processing thread (non-blocking)
                try:
                    self.process_queue.put_nowait(audio_chunk.copy())
                except queueing.Full:
                    pass  # Skip processing if busy
                
                # Try to get processed audio, otherwise passthrough
                
                try:
                    in_list = np.array([])
                    out_list = np.array([])
                    time_list = np.array([])
                    while self.output_queue.qsize() > 1:
                        in_list = np.append(in_list, self.input_queue.get_nowait(), axis=0)
                        out_list = np.append(out_list, self.output_queue.get_nowait(), axis=0)
                        time_list = np.append(time_list, [self.timing_queue.get_nowait()], axis=0)

                except queueing.Empty:
                    pass
                    # modified_out = indata   # Passthrough if no processed audio ready
                try:
                    queue_list.append(in_list)
                    queue_list.append(out_list)
                    queue_list.append(time_list)
                    queue.put_nowait(queue_list)
                except queueing.Full:
                    pass
                
                # outdata[:] = modified_out                    
            except Exception as e:
                print(f"ERROR in stream_callback: {e}")
        
        return basic_callback
    
    def stop(self):
        """Clean shutdown"""
        self.running = False
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
