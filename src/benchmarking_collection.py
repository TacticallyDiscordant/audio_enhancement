import sys, os, pathlib, importlib.util
# add project root to sys.path so 'models' package (models/__init__.py) is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
print("PYTHON:", sys.executable)
print("cwd:", pathlib.Path().resolve())
print("sys.path[0]:", sys.path[0])
p = pathlib.Path(__file__).resolve().parents[1] / "models"
print("expected models dir:", p)
print("exists:", p.exists())
print("models contents:", os.listdir(p) if p.exists() else None)
print("audiosr spec:", importlib.util.find_spec("audiosr"))
# ...existing code...
"""

import models.FLowHigh_inference as fh
import models.VAudioSR_inference as vsr
# import models.FlashSR_inference as fsr
from externals.ssr_eval import SSR_Eval_Helper 


def test_flowhigh():
    use_vocoder = True
    model_config_path = 'models/FLowHigh/model/FLowHigh_indep_adaptive_400k/config.json'
    model_path = 'models/FLowHigh/model/FLowHigh_indep_adaptive_400k/FLowHigh_indep_adaptive_400k.pt'
    ode_method = 'euler'
    up_sampling_method = 'scipy'  # scipy, librosa or torchaudio
    testee = fh.Complete_FLowHigh(
                            model_config_path=model_config_path,
                            model_path=model_path,
                            use_vocoder=use_vocoder,
                            input_sr=44100,
                            ode_method=ode_method,
                            up_sampling_method=up_sampling_method)

    test_name = f"{testee.props['model']['modelname']}_{testee.props['model']['cfm_path']}_{ode_method}_{up_sampling_method}"
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


def test_rAI_FLowHigh():
    
    testee = fh.rAI_FLowHigh(
            input_sr=44100,
            target_sr=48000
            )
    # Initialize a evaluation helper
    helper = SSR_Eval_Helper(
        testee,
        test_name='rAI_FLowHigh',  # Test name for storing the result
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

"""
def test_FlashSR():
    input_sr = 44100
    output_sr = 48000
    testee = fsr.Complete_FlashSR(input_sr,
                                    target_sr=output_sr)

    test_name = "FlashSR"
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
        save_processed_result=True)
    helper.evaluate(limit_test_nums=10, limit_test_speaker=-1)
"""

# test_flowhigh()
# test_VAudioSR()
# test_FlashSR()
test_rAI_FLowHigh()
