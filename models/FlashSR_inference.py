import torch
import torch.nn.functional as F

import numpy as np
from models.FlashSR.TorchJaekwon.Util.UtilAudio import UtilAudio
from models.FlashSR.TorchJaekwon.Util.UtilData import UtilData

from models.FlashSR.FlashSR.FlashSR import FlashSR


class Complete_FlashSR(object):

    def __init__(self,
                input_sr,
                target_sr=48000,
                audio_limit=245760,  # supposedly FlashSR can't do more than 5 seconds
                do_lowpass_filter=False,
                ldm_checkpoint='./models/weights_and_configs/FlashSR/student_ldm.pth',
                vocoder_checkpoint='./models/weights_and_configs/FlashSR/sr_vocoder.pth',
                vae_checkpoint='./models/weights_and_configs/FlashSR/vae.pth'):

        self.device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_sr = input_sr
        self.target_sr = target_sr
        self.audio_limit = audio_limit
        self.lowpass_filter = do_lowpass_filter
        self.flashsr = FlashSR(student_ldm_ckpt_path=ldm_checkpoint,
                            sr_vocoder_ckpt_path=vocoder_checkpoint,
                            autoencoder_ckpt_path=vae_checkpoint,
                            model_output_type='v_prediction',
                            beta_schedule_type='cosine')
        self.flashsr.to(self.device)


    def infer(self, audio):
        """
        Robust inference entrypoint that accepts numpy arrays or torch tensors in the
        exact form provided by externals.ssr_eval.SSR_Eval_Helper.preprocess().
        Returns a 1-D numpy array at self.target_sr.
        This implementation chunks long inputs (self.audio_limit, samples at target_sr)
        and concatenates model outputs so returned length matches the resampled input.
        """
        # normalize / coerce input, keep original sample count at input_sr
        import numpy as _np
        import torch as _torch
        import torch.nn.functional as _F
        from models.FlashSR.TorchJaekwon.Util.UtilAudio import UtilAudio

        # handle numpy input
        if isinstance(audio, _np.ndarray):
            # convert multi-channel -> mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(_np.float32)
            # normalize if values look integer-scaled
            if audio.dtype == _np.int16 or audio.dtype == _np.int32:
                audio = audio / float(_np.iinfo(audio.dtype).max)
            orig_in_samples = audio.shape[0]
        elif isinstance(audio, _torch.Tensor):
            if audio.dim() > 1:
                audio = audio.mean(dim=1)
            orig_in_samples = audio.shape[-1]
        else:
            # try to coerce (e.g., list)
            audio = _np.asarray(audio, dtype=_np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            orig_in_samples = audio.shape[0]

        # Resample / convert to torch tensor using existing utility
        # UtilAudio.take_stream should return a torch.Tensor [time] or [channels, time]
        audio = UtilAudio.take_stream(audio,
                                      input_sample_rate=self.input_sr,
                                      out_sample_rate=self.target_sr)

        # ensure torch tensor, float, and shape [batch, time]
        if not isinstance(audio, _torch.Tensor):
            audio = _torch.tensor(audio, dtype=_torch.float32)
        if not _torch.is_floating_point(audio):
            audio = audio.float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [1, time]
        elif audio.dim() == 2 and audio.shape[0] > audio.shape[1] and audio.shape[1] <= 2:
            # rare (time, channels) case -> (channels, time)
            audio = audio.transpose(0, 1)

        # move to device
        audio = audio.to(self.device)

        # record resampled length to align model output
        resampled_len = int(audio.shape[-1])

        # enforce audio length limit: trim or pad
        if resampled_len > self.audio_limit:
            audio = audio[..., : self.audio_limit]
            resampled_len = self.audio_limit
        elif resampled_len < 16:
            # extremely short guard
            pad_needed = 16 - resampled_len
            audio = _F.pad(audio, (0, pad_needed))
            resampled_len = int(audio.shape[-1])

        # Run model (FlashSR.forward expects [batch, time])
        with _torch.no_grad():
            pred = self.flashsr.forward(lr_audio=audio, lowpass_input=self.lowpass_filter)

        # pred expected as tensor [batch, time] or [time]
        if isinstance(pred, tuple):
            # some wrappers might return (audio, meta)
            pred = pred[0]
        if not isinstance(pred, _torch.Tensor):
            pred = _torch.tensor(pred, dtype=_torch.float32)

        # ensure on cpu and contiguous
        pred = pred.cpu().detach()

        # if batched -> take first item
        if pred.dim() == 2 and pred.shape[0] == 1:
            pred = pred.squeeze(0)

        # align output length to resampled_len (trim/pad)
        out_len = int(pred.shape[-1])
        if out_len > resampled_len:
            pred = pred[..., :resampled_len]
        elif out_len < resampled_len:
            pad_len = resampled_len - out_len
            pred = _F.pad(pred, (0, pad_len))

        # final numpy output: 1-D float32
        out_np = pred.squeeze().numpy().astype(_np.float32)

        return out_np

        # normalize / coerce input, keep original sample count at input_sr
        if isinstance(audio, _np.ndarray):
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(_np.float32)
            orig_in_samples = audio.shape[0]
        elif isinstance(audio, _torch.Tensor):
            if audio.dim() > 1:
                audio = audio.mean(dim=1)
            orig_in_samples = int(audio.shape[-1])
        else:
            audio = _np.asarray(audio, dtype=_np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            orig_in_samples = audio.shape[0]

        # resample to target_sr using existing util (returns tensor or ndarray)
        resampled = UtilAudio.take_stream(audio,
                                         input_sample_rate=self.input_sr,
                                         out_sample_rate=self.target_sr)

        # ensure torch Tensor [batch, time]
        if isinstance(resampled, _np.ndarray):
            resampled = _torch.from_numpy(resampled)
        if not isinstance(resampled, _torch.Tensor):
            resampled = _torch.tensor(resampled, dtype=_torch.float32)
        if resampled.dim() == 1:
            resampled = resampled.unsqueeze(0)
        elif resampled.dim() == 2 and resampled.shape[0] > resampled.shape[1] and resampled.shape[1] <= 2:
            resampled = resampled.transpose(0, 1)

        resampled = resampled.to(self.device).float()
        total_len = int(resampled.shape[-1])

        # chunking params (self.audio_limit is in samples at target_sr)
        max_chunk = int(self.audio_limit)
        outputs = []
        idx = 0

        while idx < total_len:
            end = min(idx + max_chunk, total_len)
            seg = resampled[..., idx:end]  # [1, seg_len]
            seg_len = seg.shape[-1]

            # pad segment to chunk size for model if needed
            need_pad = max_chunk - seg_len if (end - idx) < max_chunk else 0
            if need_pad > 0:
                seg = _F.pad(seg, (0, need_pad))

            with _torch.no_grad():
                pred = self.flashsr.forward(lr_audio=seg, lowpass_input=self.lowpass_filter)

            # normalize output tensor shape
            if isinstance(pred, tuple):
                pred = pred[0]
            if not isinstance(pred, _torch.Tensor):
                pred = _torch.tensor(pred, dtype=_torch.float32)
            pred = pred.cpu().detach()
            if pred.dim() == 2 and pred.shape[0] == 1:
                pred = pred.squeeze(0)

            # trim padding from last chunk
            if need_pad > 0:
                pred = pred[..., :seg_len]

            outputs.append(pred)
            idx += max_chunk

        # concatenate and ensure exact total_len
        if len(outputs) == 0:
            out = _torch.zeros((total_len,), dtype=_torch.float32)
        else:
            out = _torch.cat(outputs, dim=-1)
            if out.dim() == 2 and out.shape[0] == 1:
                out = out.squeeze(0)

        if out.shape[-1] > total_len:
            out = out[..., :total_len]
        elif out.shape[-1] < total_len:
            out = _F.pad(out, (0, total_len - out.shape[-1]))

        out_np = out.cpu().numpy().astype(_np.float32)
        return out_np

