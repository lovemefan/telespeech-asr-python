# -*- coding:utf-8 -*-
# @FileName  :infer.py
# @Time      :2024/5/31 13:58
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import argparse
import json
import logging
import os
import time
from typing import Dict, List

import kaldifeat
import torch

from telespeechasr.torch.model.data2vec_multi_model import Data2VecMultiModel
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
        logging.info(f"Loading model from {self.model_path}")
        self.model = Data2VecMultiModel()
        load_checkpoint(model_path, self.model)
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
        logging.info(f"Decoding {audio_path}")
        start_time = time.time()
        device = torch.device(device)
        wave = read_wave(audio_path)
        feats = self.mfcc(wave.cpu())
        feats = self.postprocess(feats).unsqueeze(0).to(device)

        extractor_out = self.model.modality_encoders(
            feats,
            torch.zeros(feats.shape[:2], dtype=torch.bool),
            False,
            remove_masked=False,
            clone_batch=1,
            mask_seeds=None,
            precomputed_mask=None,
        )

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        layer_results = []
        for i, blk in enumerate(self.model.blocks):
            ab = masked_alibi_bias
            if ab is not None and alibi_scale is not None:
                scale = (
                    alibi_scale[i]
                    if alibi_scale.size(0) > 1
                    else alibi_scale.squeeze(0)
                )
                ab = ab * scale.type_as(ab)

            x, lr = blk(
                x,
                padding_mask=masked_padding_mask,
                alibi_bias=ab,
            )

            layer_results.append((x, lr))

        x.transpose_(0, 1)
        model_output = self.model.proj(x)
        emissions = self.get_logits(model_output)
        emissions = emissions.transpose(0, 1).float().cpu().contiguous()
        hypos = self.viterbi_decode(emissions)

        result = self.postprocess_sentence(hypos[0][0]["tokens"])
        logging.info(f"Inference time: {time.time() - start_time}s")
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

    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    inference_processor = InferenceProcessor(
        args.model_path, args.vocab_path, device=args.device
    )
    asr_result = inference_processor.infer(args.audio_path, device=args.device)
    logging.info(asr_result)
