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
from telespeechasr.torchscript.torchscript_export import data2vec_multo_model_export


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
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Whether to quantize the model",
    )
    args = parser.parse_args()
    return args


def export_onnx(args, model):
    model_path = os.path.join(args.output_dir, f"model_export.onnx")
    torch.onnx.export(
        model,
        (torch.randn(1, 155, 40)),
        model_path,
        verbose=False,
        opset_version=11,
        input_names=["feats"],
        output_names=["logits"],
        dynamic_axes={
            "feats": {1: "T"},
            "encoder_out": {1: "T"},
        },
    )
    if args.quantize:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quant_model_path = os.path.join(
            args.output_dir, f"model_export_int8_quant.onnx"
        )
        if not os.path.exists(quant_model_path):
            quantize_dynamic(
                model_input=model_path,
                model_output=quant_model_path,
                op_types_to_quantize=["MatMul"],
                per_channel=True,
                reduce_range=False,
                weight_type=QuantType.QUInt8,
            )


if __name__ == "__main__":
    args = get_parser()
    model = Data2VecMultiModel(vocab_size=7535)
    model = load_checkpoint(args.model_path, model)
    model_export = data2vec_multo_model_export(model)
    export_onnx(args, model_export)
