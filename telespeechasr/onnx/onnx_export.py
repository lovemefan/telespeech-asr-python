# -*- coding:utf-8 -*-
# @FileName  :export.py
# @Time      :2024/6/3 15:15
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import argparse
import os

import torch

from telespeechasr.torch.model.data2vec_multi_model import Data2VecMultiModel
from telespeechasr.torch.utils.utils import load_checkpoint
from telespeechasr.torchscript.export import data2vec_multo_model_export


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output dir of model checkpoint",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    model = Data2VecMultiModel()
    model = load_checkpoint(args.model_path, model)
    model_export = data2vec_multo_model_export(model)
    torch.onnx.export(
        model_export,
        (torch.randn(1, 155, 40)),
        os.path.join(args.output_dir, f"model_export.onnx"),
        verbose=False,
        opset_version=11,
        input_names=["feats"],
        output_names=["logits"],
        dynamic_axes={
            "feats": {1: "T"},
            "encoder_out": {1: "T"},
        },
    )
