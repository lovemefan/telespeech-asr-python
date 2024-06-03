# -*- coding:utf-8 -*-
# @FileName  :infer.py.py
# @Time      :2024/6/3 15:00
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import argparse
import json
import os
from typing import Dict, List

import kaldifeat
import torch

from telespeechasr.torch.utils.utils import load_checkpoint, read_wave


class InferenceProcessor:
    def __init__(self, model_path, vocab_path=None, device: str = "cuda"):
        self.model_path = model_path
        self.vocab_path = vocab_path or os.path.join(
            os.path.dirname(__file__), "data", "vocab.json"
        )

        with open(self.vocab_path, "r") as f:
            self.vocab2id = json.load(f)
            self.id2vocab = {}
            for k, v in self.vocab2id.items():
                self.id2vocab[v] = k

        self.model = torch.jit.load(self.model_path)
        self.model.eval()
        self.model = self.model.to(device)

        opts = kaldifeat.MfccOptions()
        opts.device = torch.device("cpu")
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
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        def get_pred(e):
            toks = e.argmax(dim=-1).unique_consecutive()
            return toks[toks != 0].cpu().numpy()

        return [[{"tokens": get_pred(x), "score": 0}] for x in emissions]

    def postprocess_sentence(self, tokens):
        text = ""
        for token in tokens:
            if token in self.id2vocab:
                token = self.id2vocab[token]
                text += token
        return text

    @torch.no_grad()
    def infer(self, audio_path, device="cuda"):
        device = torch.device(device)
        wave = read_wave(audio_path)
        feats = self.mfcc(wave.cpu())
        feats = self.postprocess(feats).unsqueeze(0).to(device)

        model_output = self.model(feats)

        emissions = self.get_logits(model_output)
        emissions = emissions.transpose(0, 1).float().cpu().contiguous()
        hypos = self.viterbi_decode(emissions)

        result = self.postprocess_sentence(hypos[0][0]["tokens"])

        return result


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, required=True)
    args.add_argument("--audio_path", type=str, required=True)
    args.add_argument("--vocab_path", type=str, default=None)
    args.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"]
    )

    args = args.parse_args()

    inference_processor = InferenceProcessor(
        args.model_path, args.vocab_path, device=args.device
    )
    asr_result = inference_processor.infer(args.audio_path, device=args.device)
    print(asr_result)
