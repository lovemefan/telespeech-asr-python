# -*- coding:utf-8 -*-
# @FileName  :convert_fairseq_checkpoint.py
# @Time      :2024/7/9 09:19
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import argparse
import logging

import torch

formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
logging.basicConfig(format=formatter, level=logging.INFO)

MAPPING = {
    "w2v_encoder.proj": "proj",
    "w2v_encoder.w2v_model.modality_encoders.AUDIO.alibi_scale": "modality_encoders.alibi_scale",
    "w2v_encoder.w2v_model.modality_encoders.AUDIO.local_encoder": "modality_encoders.local_encoder",
    "w2v_encoder.w2v_model.modality_encoders.AUDIO.project_features": "modality_encoders.project_features",
    "w2v_encoder.w2v_model.modality_encoders.AUDIO.relative_positional_encoder": "modality_encoders.relative_positional_encoder",
    "w2v_encoder.w2v_model.modality_encoders.AUDIO.context_encoder": "modality_encoders.context_encoder",
    "w2v_encoder.w2v_model.blocks": "blocks",
}


def recursively_load_weights(fairseq_model):
    unused_weights = ["_ema", "modality_encoders.AUDIO.decoder.blocks"]
    new_weights = {}

    items = list(fairseq_model.items())
    for name, value in items:
        for w in unused_weights:
            if w in name:
                logging.warning(f"Unused weights: {name}")
                del fairseq_model[name]

    for name, value in fairseq_model.items():
        is_needed = True
        for key, mapped_key in MAPPING.items():
            if key in name:
                logging.info(f"Replace {name} with {name.replace(key, mapped_key)}")
                new_weights[name.replace(key, mapped_key)] = value
                is_needed = False
        if is_needed:
            new_weights[name] = value






    [print(i) for i in new_weights.keys()]
    return new_weights


def convert_telespeech_checkpoint(input, output):
    fairseq_model = torch.load(input, map_location="cpu")
    new_weights = recursively_load_weights(fairseq_model["model"])
    torch.save(new_weights, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    convert_telespeech_checkpoint(args.input, args.output)
