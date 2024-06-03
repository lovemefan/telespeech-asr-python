import math
from collections import namedtuple
from typing import Optional, Tuple

import numpy as np
import torch
from torch import functional as F
from torch import nn

MaskSeed = namedtuple("MaskSeed", ["seed", "update", "ids"])
MaskInfo = namedtuple("MaskInfo", ["x_unmasked", "mask", "ids_restore", "ids_keep"])


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
    add_masks: bool = False,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
    idc_select_ver: int = 1,  # 2 to reproduce mask_tokens_dataset
    num_mask_ver: int = 2,  # 2 to reproduce mask_tokens_dataset
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    if num_mask_ver == 1:
        all_num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * all_sz / float(mask_length)
            + np.random.rand()
        )
        all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if seed is not None and epoch is not None and indices is not None:
            seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
        else:
            seed_i = None

        rng = np.random.default_rng(seed_i)

        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else:
            sz = all_sz

        if num_mask_ver == 1:
            if padding_mask is not None:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    mask_prob * sz / float(mask_length)
                    + np.random.rand()
                )
                num_mask = max(min_masks, num_mask)
            else:
                num_mask = all_num_mask
        elif num_mask_ver == 2:
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + rng.random()
            )
            num_mask = max(min_masks, num_mask)
        else:
            raise ValueError()

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = rng.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = rng.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            if mask_type == "static":
                raise ValueError(f"this should never happens")
            else:
                lengths = [min(mask_length, sz - 1)]

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = rng.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = rng.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            if idc_select_ver == 1:
                min_len = min(lengths)
                if sz - min_len <= num_mask:
                    min_len = sz - num_mask - 1
                mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
            elif idc_select_ver == 2:
                mask_idc = rng.choice(sz, num_mask, replace=False)
            else:
                raise ValueError()

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz:
            raise ValueError(
                (
                    f"the entire sequence is masked. "
                    f"sz={sz}; mask_idc[mask_idc]; "
                    f"index={indices[i] if indices is not None else None}"
                )
            )
        mask_idcs.append(mask_idc)

    target_len = None
    if require_same_masks:
        if add_masks:
            target_len = max([len(m) for m in mask_idcs])
        else:
            target_len = min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = rng.choice(mask_idc, target_len, replace=False)

        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True

        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            num_holes = np.rint(len(masked) * mask_dropout).astype(int)
            to_drop = rng.choice(masked, num_holes, replace=False)
            mask[i, to_drop] = False

    return mask


def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == "xla"


def index_put(tensor, indices, value):
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()):
            indices = indices.unsqueeze(-1)
        if indices.size(-1) < tensor.size(-1):
            indices = indices.expand_as(tensor)
        tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
    else:
        tensor[indices] = value
    return tensor


def get_alibi(
    max_positions: int,
    attention_heads: int,
    dims: int = 1,
    distance: str = "manhattan",
):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            # 2 ** (-(2 ** -(math.log2(n) - 3))) equals 2 ** -(8/n)
            # start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            start = 2 ** -(8 / n)
            ratio = start
            return [start * ratio**i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some
        # a. This function has some good properties that only occur when
        # the input is a power of 2. To maintain that even when the number
        # of heads is not a power of 2, we use this workaround.

        # math.log2(n).is_integer() equals  n & (n - 1) == 0
        # if math.log2(n).is_integer():
        if n & (n - 1) == 0:
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    maxpos = max_positions
    attn_heads = attention_heads
    slopes = torch.Tensor(get_slopes(attn_heads))

    if dims == 1:
        # prepare alibi position linear bias. Note that wav2vec2 is non
        # autoregressive model so we want a symmetric mask with 0 on the
        # diagonal and other wise linear decreasing valuees
        pos_bias = (
            torch.abs(
                torch.arange(maxpos).unsqueeze(0) - torch.arange(maxpos).unsqueeze(1)
            )
            * -1
        )
    elif dims == 2:
        if distance == "manhattan":
            df = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
        elif distance == "euclidean":
            df = lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        n = math.sqrt(max_positions)
        assert n.is_integer(), n
        n = int(n)

        pos_bias = torch.zeros((max_positions, max_positions))

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        new_x = i * n + j
                        new_y = k * n + l
                        pos_bias[new_x, new_y] = -df(i, j, k, l)

    else:
        raise Exception(f"unsupported number of alibi dims: {dims}")

    alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * pos_bias.unsqueeze(0).expand(
        attn_heads, -1, -1
    )

    return alibi_bias


def gather_unmasked(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep,
    )


def gather_unmasked_mask(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep[..., 0],  # ignore the feature dimension
    )


def get_alibi_bias(
    batch_size,
    time_steps,
    heads,
    dtype,
    device,
    dims=1,
    distance="manhattan",
):
    target_size = heads * batch_size

    buffered = (
        get_alibi(time_steps, heads, dims=dims, distance=distance)
        .to(dtype=dtype, device=device)
        .repeat(target_size, 1, 1)
    )

    b = buffered[:target_size, :time_steps, :time_steps]
    b = b.view(batch_size, heads, time_steps, time_steps)
    return b


def masked_alibi(alibi_bias, mask_info):
    H = alibi_bias.size(1)

    orig_bias = alibi_bias

    index = mask_info.ids_keep.unsqueeze(1)[..., 0].unsqueeze(-1)
    alibi_bias = torch.gather(
        orig_bias,
        dim=-2,
        index=index.expand(-1, H, -1, mask_info.ids_restore.size(1)),
    )
    alibi_bias = torch.gather(
        alibi_bias,
        dim=-1,
        index=index.transpose(-1, -2).expand(-1, H, alibi_bias.size(-2), -1),
    )

    return alibi_bias


# adapted from MAE
def random_masking(x, mask_ratio, mask_seed: Optional[MaskSeed]):
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    generator = None
    if mask_seed is not None:
        seed = int(
            hash((mask_seed.seed, mask_seed.update, mask_seed.ids.sum().item())) % 1e6
        )
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)

    noise = torch.rand(N, L, generator=generator, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = noise.argsort(dim=1)  # ascend: small is keep, large is remove
    ids_restore = ids_shuffle.argsort(dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    x_unmasked = torch.gather(x, dim=1, index=ids_keep)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], dtype=x.dtype, device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, D)

    return MaskInfo(
        x_unmasked=x_unmasked, mask=mask, ids_restore=ids_restore, ids_keep=ids_keep
    )


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class ModalitySpecificEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        local_encoder: nn.Module,
        project_features: nn.Module,
        fixed_positional_encoder: Optional[nn.Module],
        relative_positional_encoder: Optional[nn.Module],
        context_encoder: nn.Module,
        decoder: nn.Module,
        num_extra_tokens: int = 0,
        prenet_depth: int = 8,
        model_depth: int = 16,
        local_grad_mult: float = 0.0,
        num_alibi_heads: int = 16,
        mask_noise_std: float = 0.01,
    ):
        super().__init__()

        self.local_encoder = local_encoder
        self.project_features = project_features
        self.fixed_positional_encoder = fixed_positional_encoder
        self.relative_positional_encoder = relative_positional_encoder
        self.context_encoder = context_encoder

        self.decoder = decoder
        self.get_alibi_bias = get_alibi_bias
        self.num_extra_tokens = num_extra_tokens
        self.prenet_depth = prenet_depth
        self.model_depth = model_depth
        self.local_grad_mult = local_grad_mult
        self.num_alibi_heads = num_alibi_heads
        self.mask_noise_std = mask_noise_std

        self.extra_tokens = None
        if num_extra_tokens > 0:
            self.extra_tokens = nn.Parameter(
                torch.zeros(1, num_extra_tokens, embed_dim)
            )

            nn.init.normal_(self.extra_tokens[:, 1:])

        self.alibi_scale = None
        if self.get_alibi_bias is not None:
            self.alibi_scale = nn.Parameter(
                torch.full(
                    (
                        1,
                        1,
                        num_alibi_heads,
                        1,
                        1,
                    ),
                    1.0,
                    dtype=torch.float,
                ),
                requires_grad=True,
            )

    def upgrade_state_dict_named(self, state_dict, name):
        k = f"{name}.alibi_scale"
        if k in state_dict and state_dict[k].dim() == 4:
            state_dict[k] = state_dict[k].unsqueeze(0)

        return state_dict

    def convert_padding_mask(self, x, padding_mask):
        return padding_mask

    def decoder_input(self, x, mask_info: MaskInfo, inp_drop=0.1):
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)

        num_extra = self.num_extra_tokens

        if mask_info is not None:
            num_masked = mask_info.ids_restore.shape[1] - x.shape[1] + num_extra

            mask_tokens = x.new_empty(
                x.size(0),
                num_masked,
                x.size(-1),
            ).normal_(0, self.mask_noise_std)

            x_ = torch.cat([x[:, num_extra:], mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=mask_info.ids_restore)

        else:
            x = x[:, num_extra:]

        return x, mask_info

    def local_features(self, features):
        if self.local_grad_mult > 0:
            if self.local_grad_mult == 1.0:
                x = self.local_encoder(features)
            else:
                x = GradMultiply.apply(
                    self.local_encoder(features), self.local_grad_mult
                )
        else:
            with torch.no_grad():
                x = self.local_encoder(features)

        x = self.project_features(x)
        return x

    def contextualized_features(
        self,
        x,
        padding_mask,
        mask,
        remove_masked,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):
        if padding_mask is not None:
            padding_mask = self.convert_padding_mask(x, padding_mask)

        local_features = x
        if mask and clone_batch == 1:
            local_features = local_features.clone()

        orig_B, orig_T, _ = x.shape
        pre_mask_B = orig_B
        mask_info = None

        x_pos = None
        if self.fixed_positional_encoder is not None:
            x = x + self.fixed_positional_encoder(x, padding_mask)

        if mask:
            if clone_batch > 1:
                x = x.repeat_interleave(clone_batch, 0)
                if mask_seeds is not None:
                    clone_hash = [
                        int(hash((mask_seeds.seed, ind)) % 1e10)
                        for ind in range(clone_batch - 1)
                    ]
                    clone_hash = torch.tensor([0] + clone_hash).long().view(1, -1)

                    id = mask_seeds.ids
                    id = id.repeat_interleave(clone_batch, 0)
                    id = id.view(-1, clone_batch) + clone_hash.to(id)
                    id = id.view(-1)
                    mask_seeds = MaskSeed(
                        seed=mask_seeds.seed, update=mask_seeds.update, ids=id
                    )
                if padding_mask is not None:
                    padding_mask = padding_mask.repeat_interleave(clone_batch, 0)

            x, mask_info = self.compute_mask(
                x,
                padding_mask,
                mask_seed=mask_seeds,
                apply=self.relative_positional_encoder is not None or not remove_masked,
                precomputed_mask=precomputed_mask,
            )

        if self.relative_positional_encoder is not None:
            x_pos = self.relative_positional_encoder(x)

        masked_padding_mask = padding_mask
        if mask and remove_masked:
            x = mask_info.x_unmasked
            if x_pos is not None:
                x = x + gather_unmasked(x_pos, mask_info)

            if padding_mask is not None and padding_mask.any():
                masked_padding_mask = gather_unmasked_mask(padding_mask, mask_info)
                if not masked_padding_mask.any():
                    masked_padding_mask = None
            else:
                masked_padding_mask = None

        elif x_pos is not None:
            x = x + x_pos

        alibi_bias = None
        alibi_scale = self.alibi_scale

        if self.get_alibi_bias is not None:
            alibi_bias = self.get_alibi_bias(
                batch_size=pre_mask_B,
                time_steps=orig_T,
                heads=self.num_alibi_heads,
                dtype=torch.float32,
                device=x.device,
            )

            if alibi_scale is not None:
                alibi_scale = alibi_scale.clamp_min(0)
                if alibi_scale.size(0) == 1:
                    alibi_bias = alibi_bias * alibi_scale.squeeze(0).type_as(alibi_bias)
                    alibi_scale = None

            if clone_batch > 1:
                alibi_bias = alibi_bias.repeat_interleave(clone_batch, 0)

            if mask_info is not None and remove_masked:
                alibi_bias = masked_alibi(alibi_bias, mask_info)

        if self.extra_tokens is not None:
            num = self.extra_tokens.size(1)
            x = torch.cat([self.extra_tokens.expand(x.size(0), -1, -1), x], dim=1)
            if masked_padding_mask is not None:
                # B x T
                masked_padding_mask = F.pad(masked_padding_mask, (num, 0))
            if alibi_bias is not None:
                # B x H x T x T
                alibi_bias = F.pad(alibi_bias, (num, 0, num, 0))

        x = self.context_encoder(
            x,
            masked_padding_mask,
            alibi_bias,
            alibi_scale[: self.prenet_depth] if alibi_scale is not None else None,
        )

        return {
            "x": x,
            "local_features": local_features,
            "padding_mask": masked_padding_mask,
            "alibi_bias": alibi_bias,
            "alibi_scale": alibi_scale[self.prenet_depth :]
            if alibi_scale is not None and alibi_scale.size(0) > 1
            else alibi_scale,
            "encoder_mask": mask_info,
        }

    def forward(
        self,
        features,
        padding_mask,
        mask: bool,
        remove_masked: bool,
        clone_batch: int = 1,
        mask_seeds: Optional[torch.Tensor] = None,
        precomputed_mask=None,
    ):
        x = self.local_features(features)
        return self.contextualized_features(
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch,
            mask_seeds,
            precomputed_mask,
        )

    def reset_parameters(self):
        pass

    def compute_mask(
        self,
        x,
        padding_mask,
        mask_seed: Optional[MaskSeed],
        apply,
        precomputed_mask,
        mask_prob: float = 0.5,
        mask_length: int = 3,
    ):
        if precomputed_mask is not None:
            mask = precomputed_mask
            mask_info = self.make_maskinfo(x, mask)
        else:
            B, T, C = x.shape

            if mask_prob > 0:
                if mask_length == 1:
                    mask_info = random_masking(x, mask_prob, mask_seed)
                else:
                    mask = compute_mask_indices(
                        (B, T),
                        padding_mask,
                        mask_prob,
                        mask_length,
                        min_masks=1,
                        require_same_masks=True,
                        mask_dropout=0.0,
                        add_masks=False,
                        seed=mask_seed.seed if mask_seed is not None else None,
                        epoch=mask_seed.update if mask_seed is not None else None,
                        indices=mask_seed.ids if mask_seed is not None else None,
                    )

                    mask = torch.from_numpy(mask).to(device=x.device)
                    mask_info = self.make_maskinfo(x, mask)
            else:
                mask_info = None

        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info

    def make_maskinfo(self, x, mask, shape=None):
        if shape is None:
            B, T, D = x.shape
        else:
            B, T, D = shape

        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)

        len_keep = T - mask[0].sum()

        ids_keep = ids_shuffle[:, :len_keep]

        if shape is not None:
            x_unmasked = None
        else:
            ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
            x_unmasked = torch.gather(x, dim=1, index=ids_keep)

        mask_info = MaskInfo(
            x_unmasked=x_unmasked,
            mask=mask,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
        )
        return mask_info

    def apply_mask(self, x, mask_info, mask_channel_prob=0.5, mask_channel_length=64):
        B, T, C = x.shape

        if mask_info is not None:
            mask = mask_info.mask

            num_masks = mask.sum().item()
            masks = x.new_empty(num_masks, x.size(-1)).normal_(0, self.mask_noise_std)
            x = index_put(x, mask, masks)
        if mask_channel_prob > 0:
            mask_channel = compute_mask_indices(
                (B, C),
                None,
                mask_channel_prob,
                mask_channel_length,
            )
            mask_channel = (
                torch.from_numpy(mask_channel)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x = index_put(x, mask_channel, 0)
        return x

    def remove_pretraining_modules(self, keep_decoder=False):
        if not keep_decoder:
            self.decoder = None
