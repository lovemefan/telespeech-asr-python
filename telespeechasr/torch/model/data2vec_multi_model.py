# -*- coding:utf-8 -*-
# @FileName  :data2vec_multi_model.py
# @Time      :2024/5/31 10:01
# @Author    :lovemefan
# @Email     :lovemefan@outlook.com
from functools import partial

import numpy as np
import torch
from torch import nn

from telespeechasr.torch.modules.attention import AltBlock
from telespeechasr.torch.modules.encoder import AudioEncoder, Decoder1d


class Data2VecMultiModel(nn.Module):
    def __init__(
        self,
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        embed_dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        encoder_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.1,
        post_mlp_dropout: float = 0.0,
        layer_norm_first: bool = False,
        end_of_block_targets: bool = False,
        average_top_k_layers: int = 16,
        loss_beta: float = 0.0,
        loss_scale: float = None,
        dropout_input: float = 0.0,
        start_drop_path_rate: float = 0.0,
        end_drop_path_rate: float = 0.0,
        depth: int = 16,
        skip_ema: bool = False,
        ema_decay: float = 0.9997,
        recon_loss: float = 0.0,
        vocab_size: int = 7535,
    ):
        make_layer_norm = partial(
            nn.LayerNorm, eps=norm_eps, elementwise_affine=norm_affine
        )
        super().__init__()

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                drop=encoder_dropout,
                attn_drop=attention_dropout,
                mlp_drop=activation_dropout,
                post_mlp_drop=post_mlp_dropout,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=layer_norm_first,
                ffn_targets=not end_of_block_targets,
            )

        self.alibi_biases = {}

        enc = AudioEncoder(embed_dim, make_block, make_layer_norm)
        self.modality_encoders = enc

        self.ema = None

        self.average_top_k_layers = average_top_k_layers
        self.loss_beta = loss_beta
        self.loss_scale = loss_scale

        self.dropout_input = nn.Dropout(dropout_input)

        dpr = np.linspace(start_drop_path_rate, end_drop_path_rate, depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(depth)])

        self.norm = None

        self.proj = nn.Linear(embed_dim, vocab_size)

        # if self.cfg.mae_init:
        #     self.apply(self._init_weights)
        # else:
        #     from fairseq.modules.transformer_sentence_encoder import init_bert_params
        #
        #     self.apply(init_bert_params)

        # for mod_enc in self.modality_encoders.values():
        #     mod_enc.reset_parameters()

        # if not skip_ema:
        #     # self.ema = self.make_ema_teacher(ema_decay)
        #     self.shared_decoder = (
        #         Decoder1d(embed_dim) if shared_decoder is not None else None
        #     )
        #     if self.shared_decoder is not None:
        #         self.shared_decoder.apply(self._init_weights)
        #
        #     self.recon_proj = None
        #     if recon_loss > 0:
        #         self.recon_proj = nn.Linear(embed_dim, embed_dim)

        # for pn, p in self.named_parameters():
        #     if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
        #         p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

        self.num_updates = 0

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        k = prefix + "_ema"
        if self.ema is not None:
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        elif k in state_dict:
            del state_dict[k]

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
