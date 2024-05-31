# -*- coding:utf-8 -*-
# @FileName  :encoder.py
# @Time      :2024/5/31 10:32
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
from functools import partial
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from telespeechasr.torch.modules.fp32_group_norm import Fp32GroupNorm
from telespeechasr.torch.modules.layernorm import Fp32LayerNorm, LayerNorm
from telespeechasr.torch.modules.modality_specific_encoder import (
    ModalitySpecificEncoder,
    get_alibi_bias,
)
from telespeechasr.torch.modules.same_pad import SamePad
from telespeechasr.torch.modules.transpose_last import TransposeLast


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        input_feature_ndim: int = 40,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = input_feature_ndim
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxTxC -> BxCxT
        # x = x.unsqueeze(1)
        x = x.permute([0, 2, 1])

        for conv in self.conv_layers:
            x = conv(x)

        return x


class BlockEncoder(nn.Module):
    def __init__(self, blocks, norm_layer, layer_norm_first, layerdrop, dropout):
        super().__init__()
        self.blocks = blocks
        self.norm = norm_layer
        self.layer_norm_first = layer_norm_first
        self.layerdrop = layerdrop
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x, padding_mask, alibi_bias, alibi_scale):
        if self.norm is not None and not self.layer_norm_first:
            x = self.norm(x)

        x = self.dropout(x)

        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.layerdrop == 0
                or (np.random.random() > self.layerdrop)
            ):
                ab = alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)
                x, _ = blk(x, padding_mask, ab)

        if self.norm is not None and self.layer_norm_first:
            x = self.norm(x)

        return x


class Decoder1d(nn.Module):
    def __init__(
        self,
        input_dim,
        decoder_dim: int = 768,
        decoder_groups: int = 16,
        decoder_kernel: int = 7,
        decoder_layers: int = 4,
        projection_layers: int = 1,
        projection_ratio: float = 2.0,
    ):
        super().__init__()

        def make_block(in_dim):
            block = [
                nn.Conv1d(
                    in_dim,
                    decoder_dim,
                    kernel_size=decoder_kernel,
                    padding=decoder_kernel // 2,
                    groups=decoder_groups,
                ),
                SamePad(decoder_kernel),
                TransposeLast(),
                LayerNorm(decoder_dim, elementwise_affine=False),
                TransposeLast(),
                nn.GELU(),
            ]

            return nn.Sequential(*block)

        self.blocks = nn.Sequential(
            *[
                make_block(input_dim if i == 0 else decoder_dim)
                for i in range(decoder_layers)
            ]
        )

        projs = []
        curr_dim = decoder_dim
        for i in range(projection_layers - 1):
            next_dim = int(curr_dim * projection_ratio) if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim
        projs.append(nn.Linear(curr_dim, input_dim))
        if len(projs) == 1:
            self.proj = projs[0]
        else:
            self.proj = nn.Sequential(*projs)

    def reset_parameters(self):
        for mod in self.proj.modules():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()

    def add_residual(self, x, residual, i, mask_info):
        if (
            residual is None
            or not self.decoder_cfg.decoder_residual
            or residual.size(1) != x.size(1)
        ):
            return x

        ret = x + residual

        return ret

    def forward(self, x, mask_info):
        x = x.transpose(1, 2)

        residual = x

        for i, layer in enumerate(self.blocks):
            x = layer(x)
            x = self.add_residual(x, residual, i, mask_info)
            residual = x

        x = x.transpose(1, 2)
        x = self.proj(x)
        return x


class AudioEncoder(ModalitySpecificEncoder):
    def __init__(
        self,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool = False,
        alibi_biases: Dict = {},
        feature_encoder_spec: str = "[(512, 3, 2), (512, 3, 2)]",
        input_feature_ndim: int = 40,
        extractor_mode: str = "layer_norm",
        conv_pos_depth: int = 5,
        conv_pos_width: int = 95,
        conv_pos_groups: int = 16,
        conv_pos_pre_ln: bool = False,
        start_drop_path_rate: float = 0.0,
        end_drop_path_rate: float = 0.0,
        prenet_depth: int = 8,
        prenet_layerdrop: float = 0.1,
        prenet_dropout: float = 0.0,
    ):
        self.feature_enc_layers = eval(feature_encoder_spec)
        feature_embed_dim = self.feature_enc_layers[-1][0]

        local_encoder = ConvFeatureExtractionModel(
            input_feature_ndim=input_feature_ndim,
            conv_layers=self.feature_enc_layers,
            dropout=0.0,
            mode=extractor_mode,
            conv_bias=False,
        )

        project_features = nn.Sequential(
            TransposeLast(),
            nn.LayerNorm(feature_embed_dim),
            nn.Linear(feature_embed_dim, embed_dim),
        )

        num_pos_layers = conv_pos_depth
        k = max(3, conv_pos_width // num_pos_layers)

        positional_encoder = nn.Sequential(
            TransposeLast(),
            *[
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        embed_dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=conv_pos_groups,
                    ),
                    SamePad(k),
                    TransposeLast(),
                    LayerNorm(embed_dim, elementwise_affine=False),
                    TransposeLast(),
                    nn.GELU(),
                )
                for _ in range(num_pos_layers)
            ],
            TransposeLast(),
        )

        if conv_pos_pre_ln:
            positional_encoder = nn.Sequential(LayerNorm(embed_dim), positional_encoder)

        dpr = np.linspace(
            start_drop_path_rate,
            end_drop_path_rate,
            prenet_depth,
        )
        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            prenet_layerdrop,
            prenet_dropout,
        )

        # decoder = Decoder1d(embed_dim)
        decoder = None
        alibi_bias_fn = partial(get_alibi_bias, alibi_biases=alibi_biases)

        super().__init__(
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            project_features=project_features,
            fixed_positional_encoder=None,
            relative_positional_encoder=positional_encoder,
            context_encoder=context_encoder,
            decoder=decoder,
            get_alibi_bias=alibi_bias_fn,
        )

    def convert_padding_mask(self, x, padding_mask):
        def get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
            """
            Computes the output length of the convolutional layers
            """

            def _conv_out_length(input_length, kernel_size, stride):
                return torch.floor((input_length - kernel_size) / stride + 1)

            for i in range(len(self.feature_enc_layers)):
                input_lengths = _conv_out_length(
                    input_lengths,
                    self.feature_enc_layers[i][1],
                    self.feature_enc_layers[i][2],
                )

            return input_lengths.to(torch.long)

        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = get_feat_extract_output_lengths(input_lengths)

            if padding_mask.any():
                padding_mask = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)

                # these two operations makes sure that all values
                # before the output lengths indices are attended to
                padding_mask[
                    (
                        torch.arange(padding_mask.shape[0], device=padding_mask.device),
                        output_lengths - 1,
                    )
                ] = 1
                padding_mask = (
                    1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])
                ).bool()
            else:
                padding_mask = torch.zeros(
                    x.shape[:2], dtype=torch.bool, device=x.device
                )

        return padding_mask

    def reset_parameters(self):
        super().reset_parameters()
        for mod in self.project_features.children():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()
