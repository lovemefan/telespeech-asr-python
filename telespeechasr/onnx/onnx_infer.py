# -*- coding:utf-8 -*-
# @FileName  :onnx_infer.py
# @Time      :2024/6/3 15:33
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Union

import kaldifeat
import numpy as np
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    get_available_providers,
    get_device,
)

from telespeechasr.torch.utils.utils import read_wave


class OrtInferRuntimeSession:
    def __init__(self, model_file, device_id=-1, intra_op_num_threads=4):
        device_id = str(device_id)
        sess_opt = SessionOptions()
        sess_opt.intra_op_num_threads = intra_op_num_threads
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cuda_ep = "CUDAExecutionProvider"
        cuda_provider_options = {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "true",
        }
        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        EP_list = []
        if (
            device_id != "-1"
            and get_device() == "GPU"
            and cuda_ep in get_available_providers()
        ):
            EP_list = [(cuda_ep, cuda_provider_options)]
        EP_list.append((cpu_ep, cpu_provider_options))

        if isinstance(model_file, list):
            merged_model_file = b""
            for file in sorted(model_file):
                with open(file, "rb") as onnx_file:
                    merged_model_file += onnx_file.read()

            model_file = merged_model_file
        else:
            self._verify_model(model_file)
        self.session = InferenceSession(
            model_file, sess_options=sess_opt, providers=EP_list
        )

        # delete binary of model file to save memory
        del model_file

        if device_id != "-1" and cuda_ep not in self.session.get_providers():
            warnings.warn(
                f"{cuda_ep} is not avaiable for current env, the inference part is automatically shifted to be executed under {cpu_ep}.\n"
                "Please ensure the installed onnxruntime-gpu version matches your cuda and cudnn version, "
                "you can check their relations from the offical web site: "
                "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
                RuntimeWarning,
            )

    def __call__(self, input_content: np.ndarray) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content[None, ...]))
        try:
            result = self.session.run(self.get_output_names(), input_dict)
            return result
        except Exception as e:
            raise RuntimeError("ONNXRuntime inferece failed.") from e

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(
        self,
    ):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character"):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")


class TeleSpeechAsrInferSession:
    def __init__(
        self, model_file, vocab_path=None, device_id=-1, intra_op_num_threads=4
    ):
        self.vocab_path = vocab_path or os.path.join(
            os.path.dirname(__file__), "data", "vocab.json"
        )

        with open(self.vocab_path, "r") as f:
            self.vocab2id = json.load(f)
            self.id2vocab = {}
            for k, v in self.vocab2id.items():
                self.id2vocab[v] = k

        logging.info(f"Loading model from {model_file}")
        self.session = OrtInferRuntimeSession(
            model_file, device_id=device_id, intra_op_num_threads=intra_op_num_threads
        )

        opts = kaldifeat.MfccOptions()
        opts.frame_opts.dither = 0
        opts.num_ceps = 40
        opts.mel_opts.num_bins = 40
        opts.mel_opts.low_freq = 40
        opts.mel_opts.high_freq = -200
        opts.frame_opts.snip_edges = False
        self.mfcc = kaldifeat.Mfcc(opts)
        self.eps = 1e-5

        self.blank_weight = 0.0
        self.blank_mode = "add"

    def postprocess(self, feats):
        assert feats.dim() == 2, feats.dim()
        m = feats.mean(dim=0)
        std = feats.std(dim=0)
        feats = (feats - m) / (std + self.eps)
        return feats

    def get_logits(self, logits):
        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., 0] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., 0] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        return logits

    def viterbi_decode(
        self,
        emissions: np.ndarray,
    ) -> List[List[Dict[str, np.ndarray]]]:
        def get_pred(e):
            toks = e.argmax(-1)
            return toks[toks != 0]

        return [[{"tokens": get_pred(x), "score": 0}] for x in emissions]

    def postprocess_sentence(self, tokens):
        text = ""
        for token in tokens:
            if token in self.id2vocab:
                token = self.id2vocab[token]
                text += token
        return text

    def infer(self, audio_path):
        wave = read_wave(audio_path)
        feats = self.mfcc(wave.cpu())
        feats = self.postprocess(feats)[None, ...].cpu().numpy()

        logging.info("Decoding ...")
        start_time = time.time()
        model_output = self.session(feats)
        emissions = self.get_logits(model_output)
        emissions = emissions[0].transpose((1, 0, 2))
        hypos = self.viterbi_decode(emissions)
        result = self.postprocess_sentence(hypos[0][0]["tokens"])
        logging.info(f"Inference time: {time.time() - start_time:.4}s")

        return result


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, required=True)
    args.add_argument("--audio_path", type=str, required=True)
    args.add_argument("--vocab_path", type=str, default=None)
    args.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"]
    )

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = args.parse_args()
    model = TeleSpeechAsrInferSession(args.model_path, args.vocab_path)
    asr_result = model.infer(args.audio_path)
    logging.info(asr_result)
