"""
Modelling
"""
import os
import re
import shutil
import time
import glob
from collections import defaultdict
from pathlib import Path
import json
import inspect
import logging
import math
import functools
from typing import *

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertEncoder,
)
from transformers.activations import get_activation

from diff.layers import modulate, gelu
# from transformers.optimization import get_linear_schedule_with_warmup
from diff.ipa import InvariantPointAttention


from diff import beta_schedules
from diff.ema import ExponentialMovingAverage
from diff.geometry import frames_torsions_to_atom14
from diff.sampling import p_sample_loop
from diff.utils import get_offsets
from diff.wrapper import Wrapper
from .rigid_utils import Rigid, Rotation

LR_SCHEDULE = Optional[Literal["OneCycleLR", "LinearWarmup"]]
TIME_ENCODING = Literal["gaussian_fourier", "sinusoidal"]
LOSS_KEYS = Literal["l1", "smooth_l1"]
DECODER_HEAD = Literal["mlp", "linear"]


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    Built primarily for score-based models.

    Source:
    https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, embed_dim: int, scale: float = 2 * torch.pi):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        w = torch.randn(embed_dim // 2) * scale
        assert w.requires_grad == False
        self.register_buffer("W", w)

    def forward(self, x: torch.Tensor):
        """
        takes as input the time vector and returns the time encoding
        time (x): (batch_size, )
        output  : (batch_size, embed_dim)
        """
        if x.ndim > 1:
            x = x.squeeze()
        elif x.ndim < 1:
            x = x.unsqueeze(0)
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        embed = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return embed


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Positional embeddings
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        # half_dim shape
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # outer product (batch, 1) x (1, half_dim) -> (batch x half_dim)
        embeddings = time[:, None] * embeddings[None, :]
        # sin and cosine embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PositionalEncoding(nn.Module):
    """
    Positional embedding for BERT.
    Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        assert len(x.shape) == 3
        orig_shape = x.shape
        # x is a tensor of shape (batch_size, seq_len, embedding_dim)
        # permute to be (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)
        x += self.pe[: x.size(0)]
        # permute back to (batch_size, seq_len, embedding_dim)
        x = x.permute(1, 0, 2)
        assert x.shape == orig_shape, f"{x.shape} != {orig_shape}"
        return self.dropout(x)

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MLA(nn.Module):
    def __init__(self, d_model=384, n_heads=16, kv_rank=32, rope_dim=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_rank = kv_rank
        self.rope_dim = rope_dim

       
        self.W_kv_a = nn.Linear(d_model, kv_rank)
        self.W_kv_b = nn.Linear(kv_rank, 2 * n_heads * self.head_dim)
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim)
        self.register_buffer('freqs', self._precompute_freqs())
        self.out_proj = nn.Linear(d_model, d_model)

    def _precompute_freqs(self):
        theta = 1.0 / (10000 ** (torch.arange(0, self.rope_dim // 2, 1).float() / (self.rope_dim // 2)))
        t = torch.arange(512, device=theta.device)
        freqs = torch.outer(t, theta)
        return torch.polar(torch.ones_like(freqs), freqs)

    def _apply_rope(self, x):
        B, L, H, D = x.shape
        x_rot = x[..., :self.rope_dim]
        x_pass = x[..., self.rope_dim:]

        x_complex = torch.view_as_complex(
            x_rot.reshape(B, L, H, -1, 2).contiguous()
        )
        rotated = x_complex * self.freqs[:L].unsqueeze(0).unsqueeze(2)
        x_rot = torch.view_as_real(rotated).reshape(B, L, H, -1)

        return torch.cat([x_rot, x_pass], dim=-1)

    def forward(self, x, mask=None):
        B, L, _ = x.shape

        kv_latent = self.W_kv_a(x)  # [B, L, r]
        k, v = self.W_kv_b(kv_latent).chunk(2, dim=-1)  # [B, L, h*d_h]
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = self._apply_rope(q.transpose(1, 2)).transpose(1, 2)  # [B, h, L, d_h]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.type(torch.bool)
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, v).transpose(1, 2)
        context = context.reshape(B, L, self.d_model)

        return self.out_proj(context)

class TPA(nn.Module):
    def __init__(self, d_model, n_heads, rank_ratio=0.25):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rank = max(n_heads, int(d_model * rank_ratio))
        assert self.rank % n_heads == 0, f"rank({self.rank}) must be divisible by n_heads({n_heads})"
        self.rank_per_head = self.rank // n_heads

        self.Wq = nn.Linear(d_model, self.rank, bias=False)
        self.Wk = nn.Linear(d_model, self.rank, bias=False)
        self.Wv = nn.Linear(d_model, self.rank, bias=False)

        self.U = nn.Parameter(torch.Tensor(n_heads, self.head_dim, self.rank_per_head))
        self.V = nn.Parameter(torch.Tensor(n_heads, self.rank_per_head, self.head_dim))

        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

        # RoPE
        self.register_buffer('freqs', self._precompute_freqs())

    def _precompute_freqs(self):
        theta = 1.0 / (10000 ** (torch.arange(0, self.head_dim // 2, 1).float() / (self.head_dim // 2)))
        seq_len = 512
        t = torch.arange(seq_len, device=theta.device)
        freqs = torch.outer(t, theta)
        return torch.polar(torch.ones_like(freqs), freqs)

    def _apply_rope(self, x):
        B, L, H, D = x.shape  # [batch, seq_len, heads, head_dim]
        x_complex = torch.view_as_complex(x.reshape(B, L, H, D // 2, 2).contiguous())
        rotated = x_complex * self.freqs[:L].unsqueeze(0).unsqueeze(2)
        return torch.view_as_real(rotated).reshape(B, L, H, D)

    def forward(self, x, mask):
        # (batch, L, d_model)
        B, L, _ = x.shape

        q_low = self.Wq(x).view(B, L, self.n_heads, self.rank_per_head)
        k_low = self.Wk(x).view(B, L, self.n_heads, self.rank_per_head)
        v_low = self.Wv(x).view(B, L, self.n_heads, self.rank_per_head)

        q = torch.einsum('hdr,blhr->blhd', self.U, q_low)  # [B,L,H,D]
        k = torch.einsum('hdr,blhr->blhd', self.U, k_low)

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        mask = mask.type(torch.bool)
        attn = torch.einsum('blhd,bmhd->bhlm', q, k) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.softmax(attn, dim=-1)

        output = torch.einsum('bhlm,bmhr->blhr', attn, v_low)
        output = torch.einsum('blhr,hrd->blhd', output, self.V)

        return output.reshape(B, L, self.d_model)

class IPALayer(nn.Module):
    """Transformer layer block."""

    def __init__(self, embed_dim, ffn_embed_dim, mha_heads, dropout=0.0,
                 use_rotary_embeddings=False, ipa_args=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.mha_heads = mha_heads
        self.inf = 1e5
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args)

    def _init_submodules(self, add_bias_kv=False, dropout=0.0, ipa_args=None):
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 6 * self.embed_dim, bias=True)
        )

        self.ipa_norm = nn.LayerNorm(self.embed_dim)
        self.ipa = InvariantPointAttention(**ipa_args)

        self.mha_l = TPA(
            self.embed_dim,
            self.mha_heads)

        self.mha_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t, mask=None, frames=None):
        shift_msa_l, scale_msa_l, gate_msa_l, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1)
        x = x + self.ipa(self.ipa_norm(x), frames, frame_mask=mask)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_l, scale_msa_l)
        x = self.mha_l(x, mask=mask)
        x = residual + gate_msa_l.unsqueeze(1) * x

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self.fc2(gelu(self.fc1(x)))
        x = residual + gate_mlp.unsqueeze(1) * x

        return x


class BertEmbeddings(nn.Module):
    """
    Adds in positional embeddings if using absolute embeddings, adds layer norm and dropout
    """

    def __init__(self, config):
        super().__init__()
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "absolute":
            # self.position_embeddings = nn.Embedding(
            #     config.max_position_embeddings, config.hidden_size
            # )
            # self.register_buffer(
            #     "position_ids",
            #     torch.arange(config.max_position_embeddings).expand((1, -1)),
            # )
            self.register_buffer('pos_embed',
                                 nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.hidden_size),
                                              requires_grad=False))
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1],
                                                          np.arange(config.max_position_embeddings))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        assert position_ids is not None, "`position_ids` must be defined"
        embeddings = input_embeds
        if self.position_embedding_type == "absolute":
            # position_embeddings = self.position_embeddings(position_ids).unsqueeze(1).expand(-1,input_embeds.shape[1], -1, -1)
            # embeddings += position_embeddings
            embeddings += self.pos_embed

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AnglesPredictor(nn.Module):
    """
    Predict angles from the embeddings. For BERT, the MLM task is done using an
    architecture like
    d_model -> dense -> d_model -> activation -> layernorm -> dense -> d_output
    https://github.com/huggingface/transformers/blob/v4.21.1/src/transformers/models/bert/modeling_bert.py#L681

    activation should be given as nn.ReLU for example -- NOT nn.ReLU()
    """

    def __init__(
        self,
        d_model: int,
        d_out: int = 4,
        activation: Union[str, nn.Module] = "gelu",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.dense1 = nn.Linear(d_model, d_model)

        if isinstance(activation, str):
            self.dense1_act = get_activation(activation)
        else:
            self.dense1_act = activation()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)

        self.dense2 = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.dense1_act(x)
        x = self.layer_norm(x)
        x = self.dense2(x)
        return x


class BertForDiffusionBase(BertPreTrainedModel, Wrapper):
    """
    BERT designed to be used with continuous inputs instead of tokens

    Reference: https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/bert/modeling_bert.py#L870

    Decoder: linear = single linear decoding of per-position embeddings
             mlp = two-layer MLP to decode per-position embeddings

    This is the base model object and does _not_ include the pytorch lightning code
    """

    # Define loss functions and their wrapped angular versions
    nonangular_loss_fn_dict = {
        "l1": F.l1_loss,
        "smooth_l1": F.smooth_l1_loss,
    }
    # To have legacy models still work with these
    loss_autocorrect_dict = {
        "radian_l1_smooth": "smooth_l1",
    }

    def __init__(
        self,
        config,
        args,
        out_dim: int = 21,
        ft_is_angular: List[bool] = [False, True, True, True],
        ft_names: Optional[List[str]] = None,
        time_encoding: TIME_ENCODING = "gaussian_fourier",
        decoder: DECODER_HEAD = "mlp",
    ) -> None:
        """
        dim should be the dimension of the inputs
        """
        super().__init__(config, args)
        self.config = config
        self.args = args
        if self.config.is_decoder:
            raise NotImplementedError
        self.ft_is_angular = ft_is_angular
        n_inputs = out_dim
        if self.args.tps_condition:
            n_inputs = 28
        self.n_inputs = n_inputs
        self.latent_dim = n_inputs

        if ft_names is not None:
            self.ft_names = ft_names
        else:
            self.ft_names = [f"ft{i}" for i in range(n_inputs)]
        assert (
            len(self.ft_names) == n_inputs
        ), f"Got {len(self.ft_names)} names, expected {n_inputs}"

        # Needed to project the low dimensional input to hidden dim
        self.inputs_to_hidden_dim = nn.Linear(
            in_features=n_inputs, out_features=config.hidden_size
        )
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        # Set up the network to project token representation to our four outputs
        if decoder == "linear":
            self.token_decoder = nn.Linear(config.hidden_size, n_inputs)
        elif decoder == "mlp":
            self.token_decoder = AnglesPredictor(config.hidden_size, n_inputs)
        else:
            raise ValueError(f"Unrecognized decoder: {decoder}")

        # Set up the time embedder
        if time_encoding == "gaussian_fourier":
            self.time_embed = GaussianFourierProjection(config.hidden_size)
        elif time_encoding == "sinusoidal":
            self.time_embed = SinusoidalPositionEmbeddings(config.hidden_size)
        else:
            raise ValueError(f"Unknown time encoding: {time_encoding}")
        pl.utilities.rank_zero_info(f"Using time embedding: {self.time_embed}")

        if not hasattr(args, 'ema'):
            args.ema = False
        if args.ema:
            self.ema = ExponentialMovingAverage(
                model=self.model, decay=args.ema_decay
            )
            self.cached_weights = None
        self.T = self.args.num_timesteps

        # Initialize weights and apply final processing
        self.init_weights()

        # Epoch counters and timers
        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()
        betas = beta_schedules.get_variance_schedule(self.args.beta_schedule, self.args.num_timesteps)
        self.alpha_beta_terms = beta_schedules.compute_alphas(betas)
        ipa_args = {
            'c_s': args.embed_dim,
            'c_z': 0,
            'c_hidden': args.ipa_head_dim,
            'no_heads': args.ipa_heads,
            'no_qk_points': args.ipa_qk,
            'no_v_points': args.ipa_v,
            'dropout': args.dropout,
        }
        self.ipa_layers = nn.ModuleList(
            [
                IPALayer(
                    embed_dim=args.embed_dim,
                    ffn_embed_dim=4 * args.embed_dim,
                    mha_heads=args.mha_heads,
                    dropout=args.dropout,
                    use_rotary_embeddings=not args.no_rope,
                    ipa_args=ipa_args
                )
                for _ in range(args.num_layers)
            ]
        )
        for block in self.ipa_layers:
            nn.init.constant_(block.ipa.linear_out.weight, 0)
            nn.init.constant_(block.ipa.linear_out.bias, 0)

        # if not self.args.no_aa_emb:
        #     self.aatype_to_emb = nn.Embedding(21, args.embed_dim)

    def run_ipa(
            self,
            t,
            mask,
            start_frames,
            aatype=None,
            x_d=None
    ):

        B, L = mask.shape
        x = torch.zeros(B, L, self.args.embed_dim, device=mask.device)
        # if aatype is not None and not self.args.no_aa_emb:
        #     x = x + self.aatype_to_emb(aatype)
        for layer in self.ipa_layers:
            x = layer(x, t, mask, frames=start_frames)
        return x

    def configure_optimizers(self):
        cls = torch.optim.AdamW if self.args.adamW else torch.optim.Adam
        optimizer = cls(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr,
        )
        return optimizer

    def prep_batch(self, batch):

        if self.args.no_frames:
            return self.prep_batch_no_frames(batch)

        # if self.args.hyena:
        if 'latents' in batch:
            return self.prep_hyena_batch(batch)

        rigids = Rigid(
            trans=batch['trans'],
            rots=Rotation(rot_mats=batch['rots'])
        )  # B, T, L  (1, 100, 4)
        B, T, L = rigids.shape   # (1,100,4)


        if self.args.no_offsets:
            offsets = rigids.to_tensor_7()
        else:
            offsets = get_offsets(rigids[:, 0:1], rigids)  # B, T, L, 7
        #### make sure the quaternions have real part
        offsets[..., :4] *= torch.where(offsets[:, :, :, 0:1] < 0, -1, 1)

        frame_loss_mask = batch['mask'].unsqueeze(-1).expand(-1, -1, 7)  # B, L, 7  (1,4,1 > 1,4,7)
        torsion_loss_mask = batch['torsion_mask'].unsqueeze(-1).expand(-1, -1, -1, 2).reshape(B, L, 14)

        if self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn:
            offsets_r = get_offsets(rigids[:, -1:], rigids)
            offsets_r[..., :4] *= torch.where(offsets_r[:, :, :, 0:1] < 0, -1, 1)
            offsets = torch.cat([offsets, offsets_r], -1)
            frame_loss_mask = torch.cat([frame_loss_mask, frame_loss_mask], -1)


        latents = torch.cat([offsets, batch['torsions'].view(B, T, L, 14)], -1)   # 21/28

        if self.args.supervise_all_torsions:
            torsion_loss_mask = torch.ones_like(torsion_loss_mask)
        elif self.args.supervise_no_torsions:
            torsion_loss_mask = torch.zeros_like(torsion_loss_mask)

        loss_mask = torch.cat([frame_loss_mask, torsion_loss_mask], -1)
        loss_mask = loss_mask.unsqueeze(1).expand(-1, T, -1, -1)

        ########
        cond_mask = torch.zeros(B, T, L, dtype=int, device=offsets.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
        if self.args.tps_condition:
            cond_mask[:, 0] = cond_mask[:, -1] = 1
        if self.args.cond_interval:
            cond_mask[:, ::self.args.cond_interval] = 1

        aatype_mask = torch.ones_like(batch['seqres'])
        ########

        return {
            'rigids': rigids,       #(1,100,4)
            'latents': latents,     #(1,100,4, 28)
            'loss_mask': loss_mask,  #(1,100,4, 28)
            'model_kwargs': {
                'start_frames': rigids[:, 0],  #(1,4)
                'end_frames': rigids[:, -1],
                'mask': batch['mask'].unsqueeze(1).expand(-1, T, -1),
                'aatype': torch.where(aatype_mask.bool(), batch['seqres'], 20),
                'x_cond': torch.where(cond_mask.unsqueeze(-1).bool(), latents, 0.0),
                'x_cond_mask': cond_mask,
            }
        }

    def mean_flat(x, mask):
        """
        Take the mean over all non-batch dimensions.
        """
        return torch.sum(x * mask, dim=list(range(1, len(x.size())))) / torch.sum(mask, dim=list(range(1, len(x.size()))))

    def general_step(self, batch, stage='train'):
       pass
    def inference(self, batch):

        prep = self.prep_batch(batch)
        rigids = prep['rigids']
        B, T, L = rigids.shape

        zs = torch.randn(B, T, L, self.latent_dim, device=self.device)  # self.latent_dim = 21
        samples = p_sample_loop(self, [L]*B, zs, self.T, self.alpha_beta_terms["betas"], prep)
        offsets = samples[..., :7]  # (1,100,4,28)
        if self.args.tps_condition or self.args.inpainting:
            torsions = samples[..., 14:28]
            logits = samples[..., -20:]
        else:
            torsions = samples[..., 7:21]
            logits = samples[..., -20:]
        frames = rigids[:, 0:1].compose(Rigid.from_tensor_7(offsets, normalize_quats=True))  # (1,100,4) (B, T, L)
        torsions = torsions.reshape(B, T, L, 7, 2)
        atom14 = frames_torsions_to_atom14(frames, torsions.view(B, T, L, 7, 2),
                                           batch['seqres'][:, None].expand(B, T, L))
        aa_out = batch['seqres'][:, None].expand(B, T, L)
        return atom14, aa_out

    @classmethod
    def from_dir(
        cls,
        dirname: str,
        ft_is_angular: Optional[Sequence[bool]] = None,
        load_weights: bool = True,
        idx: int = -1,
        best_by: Literal["train", "valid"] = "valid",
        copy_to: str = "",
        **kwargs,
    ):
        """
        Builds this model out from directory. Legacy mode is for loading models
        before there were separate folders for training and validation best models.
        idx indicates which model to load if multiple are given
        """
        train_args_fname = os.path.join(dirname, "training_args.json")
        with open(train_args_fname, "r") as source:
            train_args = json.load(source)
        config = BertConfig.from_json_file(os.path.join(dirname, "config.json"))

        # Handles the case where we repurpose the time encoding for seq len encoding in the AR model
        time_encoding_key = (
            "time_encoding" if "time_encoding" in train_args else "seq_len_encoding"
        )
        model_args = dict(
            config=config,
            ft_is_angular=ft_is_angular,
            time_encoding=train_args[time_encoding_key],
            decoder=train_args["decoder"],
            # lr=train_args["lr"],
            # loss=train_args["loss"],
            # l2=train_args["l2_norm"],
            # l1=train_args["l1_norm"],
            # circle_reg=train_args["circle_reg"],
            # lr_scheduler=train_args["lr_scheduler"],
            **kwargs,
        )

        if load_weights:
            epoch_getter = lambda x: int(
                re.findall(r"epoch=[0-9]+", os.path.basename(x)).pop().split("=")[-1]
            )
            subfolder = f"best_by_{best_by}"
            # Sort checkpoints by epoch -- last item is latest epoch
            ckpt_names = sorted(
                glob.glob(os.path.join(dirname, "models", subfolder, "*.ckpt")),
                key=epoch_getter,
            )
            logging.info(f"Found {len(ckpt_names)} checkpoints")
            ckpt_name = ckpt_names[idx]
            logging.info(f"Loading weights from {ckpt_name}")
            if hasattr(cls, "load_from_checkpoint"):
                # Defined for pytorch lightning module
                retval = cls.load_from_checkpoint(
                    checkpoint_path=ckpt_name, **model_args
                )
            else:
                retval = cls(**model_args)
                loaded = torch.load(ckpt_name, map_location=torch.device("cpu"))
                retval.load_state_dict(loaded["state_dict"])
        else:
            retval = cls(**model_args)
            logging.info(f"Loaded unitialized model from {dirname}")

        # If specified, copy out the requisite files to the given directory
        if copy_to:
            logging.info(f"Copying minimal model file set to: {copy_to}")
            os.makedirs(copy_to, exist_ok=True)
            copy_to = Path(copy_to)
            with open(copy_to / "training_args.json", "w") as sink:
                json.dump(train_args, sink)
            config.save_pretrained(copy_to)
            if load_weights:
                # Create the direcotry structure
                ckpt_dir = copy_to / "models" / subfolder
                os.makedirs(ckpt_dir, exist_ok=True)
                shutil.copyfile(ckpt_name, ckpt_dir / os.path.basename(ckpt_name))

        return retval

    def forward(
        self,
        inputs: torch.Tensor,
        timestep: torch.Tensor,  # Tensor of shape batch_length with time indices
        mask: torch.Tensor,
        attention_mask: torch.Tensor,
        start_frames: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_shape = inputs.size()
        batch_size, num_frames, seq_length, *_ = input_shape
        logging.debug(f"Detected batch {batch_size} and seq length {seq_length}")

        assert attention_mask is not None

        # If position IDs are not given, auto-generate them
        if position_ids is None:
            # [1, seq_length]
            position_ids = (
                torch.arange(
                    seq_length,
                )
                .expand(batch_size, -1)
                .type_as(timestep)
            )

        # attention_mask = attention_mask.reshape(batch_size, -1)
        attention_mask = attention_mask.reshape(-1, seq_length)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads. This code is taken
        # from hugggingface modeling_utils
        assert (
            attention_mask.dim() == 2
        ), f"Attention mask expected in shape (batch_size, seq_length), got {attention_mask.shape}"
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.type_as(attention_mask)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # msk = torch.ones(size=(self.config.num_attention_heads,))
        # msk = msk.type_as(inputs)
        # head_mask = self.get_head_mask(msk, self.config.num_hidden_layers)

        assert len(inputs.shape) == 4  # batch_size, num_frames, seq_length, features
        inputs_upscaled = self.inputs_to_hidden_dim(inputs)  # Batch *  num_frames * seq_len * dim

        # Pass through embeddings
        inputs_upscaled = self.embeddings(inputs_upscaled, position_ids=position_ids)

        # timestep is (batch, 1), squeeze to (batch,)
        # embedding gets to (batch, embed_dim) -> unsqueee to (batch, 1, dim)
        time_encoded = self.time_embed(timestep.squeeze(dim=-1))[:, None, None, :].expand(-1, num_frames, seq_length, -1)
        # inputs_with_time = (inputs_upscaled + time_encoded).reshape(batch_size * num_frames , seq_length, -1)
        inputs_with_time = inputs_upscaled + self.run_ipa(time_encoded[:, 0, 0], mask[:, 0], start_frames)[:, None]
        inputs_with_time = inputs_with_time.reshape(batch_size * num_frames , seq_length, -1)

        # (2 ,5, 197, 384)  (2, 384)
        # (1,100,4,384) （1， 4， 384）
        encoder_outputs = self.encoder(
            inputs_with_time,
            attention_mask=extended_attention_mask,
            # head_mask=head_mask,
            output_attentions=output_attentions,
            
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # s_att_weight = encoder_outputs.attentions[-1]
        # s_att_weight  = s_att_weight.detach().cpu().numpy()[:5, ...]

        # np.save('bert_attention_weights_1hpv.npy', s_att_weight)

        sequence_output = encoder_outputs[0]
        per_token_decoded = self.token_decoder(sequence_output).reshape(batch_size, num_frames, seq_length, -1)
        return per_token_decoded


