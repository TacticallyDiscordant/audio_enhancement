import sys, os, pathlib, importlib.util
# add project root to sys.path so 'models' package (models/__init__.py) is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import models.FLowHigh_inference as fh
import models.VAudioSR_inference as vsr
import models.FlashSR_inference as flash
from externals.ssr_eval import SSR_Eval_Helper, BasicTestee


class BasicTestee(BasicTestee):
    def __init__(self, input_sr=44100, target_sr=48000) -> None:
        super().__init__()
        self.input_sr = input_sr
        self.output_sr = target_sr

    def timed_infer(self, audio):
        t1 = time.time()
        output = audio
        t2 = time.time()
        inference_time = t2-t1
        inference_time_per_second = inference_time/ (audio.shape[0]/self.input_sr)
        return output, {'inference_speed': inference_time_per_second}


def test():
    testee = BasicTestee()
    # Initialize a evaluation helper
    helper = SSR_Eval_Helper(
        testee,
        test_name="unprocessed",  # Test name for storing the result
        input_sr=44100,  # The sampling rate of the input x in the 'infer' function
        output_sr=44100,  # The sampling rate of the output x in the 'infer' function
        evaluation_sr=48000,  # The sampling rate to calculate evaluation metrics.
        setting_fft={
            "cutoff_freq": [
                12000
            ],  # The cutoff frequency of the input x in the 'infer' function
        },
        save_processed_result=True
    )
    # Perform evaluation
    helper.evaluate(limit_test_nums=10, limit_test_speaker=-1)


def test_rAI_FLowHigh():
    ode_method = 'euler'
    architecture = "transformer"
    cfm = 'basic'
    if cfm == 'basic':
        basic = True

    testee = fh.rAI_FLowHigh(
            input_sr=44100,
            target_sr=48000,
            ode_method=ode_method,
            architecture=architecture,
            basic=basic
            )
    # Initialize a evaluation helper
    helper = SSR_Eval_Helper(
        testee,
        test_name= f'rAI_FLowHigh_{cfm}_{ode_method}_{architecture}',  # Test name for storing the result
        input_sr=44100,  # The sampling rate of the input x in the 'infer' function
        # output_sr=44100,  # The sampling rate of the output x in the 'infer' function
        output_sr=48000,
        evaluation_sr=48000,  # The sampling rate to calculate evaluation metrics.
        setting_fft={
            "cutoff_freq": [
                12000
            ],  # The cutoff frequency of the input x in the 'infer' function
        },
        save_processed_result=True
    )
    # Perform evaluation
    helper.evaluate(limit_test_nums=10, limit_test_speaker=-1)


def test_VAudioSR():
    input_sr = 44100
    output_sr = 48000
    testee = vsr.Predictor()

    testee.setup(model_name="speech",
                device="cpu",
                input_sr=input_sr,
                output_sr=output_sr)
    test_name = f"audiosr_{testee.model_name}"

    # Initialize a evaluation helper
    helper = SSR_Eval_Helper(
        testee,
        test_name=test_name,  # Test name for storing the result
        input_sr=44100,  # The sampling rate of the input x in the 'infer' function
        # output_sr=44100,  # The sampling rate of the output x in the 'infer' function
        output_sr=48000,
        evaluation_sr=48000,  # The sampling rate to calculate evaluation metrics.
        setting_fft={
            "cutoff_freq": [
                12000
            ],  # The cutoff frequency of the input x in the 'infer' function
        },
        save_processed_result=True
    )
    # Perform evaluation
    helper.evaluate(limit_test_nums=10, limit_test_speaker=-1)

def test_FlashSR():
    input_sr = 44100
    output_sr = 48000
    testee = flash.FlashSR(#model_path='./models/weights_and_configs/FlashSR/model.onnx',
                            model_path='./models/weights_and_configs/FlashSR/upsampler.pth',
                                input_sr=input_sr,
                                target_sr=output_sr)
    helper = SSR_Eval_Helper(
        testee,
        test_name='FlashSR',  # Test name for storing the result
        input_sr=44100,  # The sampling rate of the input x in the 'infer' function
        # output_sr=44100,  # The sampling rate of the output x in the 'infer' function
        output_sr=48000,
        evaluation_sr=48000,  # The sampling rate to calculate evaluation metrics.
        setting_fft={
            "cutoff_freq": [
                12000
            ],  # The cutoff frequency of the input x in the 'infer' function
        },
        save_processed_result=True
    )
    # Perform evaluation
    helper.evaluate(limit_test_nums=10, limit_test_speaker=-1)

test()
# test_VAudioSR()
# test_FlashSR()
# test_rAI_FLowHigh()
