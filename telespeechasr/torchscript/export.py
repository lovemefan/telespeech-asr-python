# -*- coding:utf-8 -*-
# @FileName  :export.py
# @Time      :2024/6/3 09:38
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
import argparse
import os

import torch
from torch import nn

from telespeechasr.torch.model.data2vec_multi_model import Data2VecMultiModel
from telespeechasr.torch.utils.utils import load_checkpoint


class data2vec_multo_model_export(nn.Module):
    def __init__(self, model: Data2VecMultiModel):
        super().__init__()
        self.model = model

    def forward(self, feats):
        x = self.model.modality_encoders.local_features(feats)
        orig_B, orig_T, _ = x.shape
        x_pos = self.model.modality_encoders.relative_positional_encoder(x)
        x = x + x_pos
        alibi_bias = self.model.modality_encoders.get_alibi_bias(
            batch_size=1,
            time_steps=orig_T,
            heads=self.model.modality_encoders.num_alibi_heads,
            dtype=torch.float32,
            device=x.device,
        )
        alibi_scale = self.model.modality_encoders.alibi_scale.clamp_min(0)
        alibi_bias = alibi_bias * alibi_scale.squeeze(0).type_as(alibi_bias)
        x = self.model.modality_encoders.context_encoder(
            x,
            None,
            alibi_bias,
            None,
        )

        for i, blk in enumerate(self.model.blocks):
            x, lr = blk(
                x,
                padding_mask=None,
                alibi_bias=alibi_bias,
            )

        x.transpose_(0, 1)
        model_output = self.model.proj(x)
        return model_output


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
    model_export = torch.jit.trace(model_export, (torch.randn(1, 155, 40)))
    torch.jit.save(model_export, os.path.join(args.output_dir, f"model_export.pt"))
