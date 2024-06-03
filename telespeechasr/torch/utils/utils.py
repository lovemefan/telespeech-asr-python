# -*- coding:utf-8 -*-
# @FileName  :utils.py
# @Time      :2024/5/31 14:06
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
from collections import OrderedDict

import numpy as np
import soundfile as sf
import torch


def load_checkpoint(checkpoint_path, model, device=torch.device("cpu")):
    """Load checkpoint from disk.
    Args:
        checkpoint_path (str): Path to checkpoint.
        model (nn.Module): Model to load.
        device (torch.device): Device to load.
    Returns:
        dict: Checkpoint loaded.
    """
    with open(checkpoint_path, "rb") as f:
        state = torch.load(f)

    model.load_state_dict(state)
    return model


def read_wave(filename) -> torch.Tensor:
    """Read a wave file and return it as a 1-D tensor.

    Note:
      You don't need to scale it to [-32768, 32767].
      We use scaling here to follow the approach in Kaldi.

    Args:
      filename:
        Filename of a sound file.
    Returns:
      Return a 1-D tensor containing audio samples.
    """
    with sf.SoundFile(filename) as sf_desc:
        sampling_rate = sf_desc.samplerate
        assert sampling_rate == 16000
        data = sf_desc.read(dtype=np.float32, always_2d=False)
    data *= 32768
    return torch.from_numpy(data)
