# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Perceiver architecture and components."""

import abc
import math

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import diotima.perceiver.io_processors as io_processors
import diotima.perceiver.position_encoding as position_encoding


#  -----------------------------------------------------------
#  ----------------------  Primitives  -----------------------
#  -----------------------------------------------------------


def attend(q, k, v, dropout_prob=0.0, attention_mask=None):
    """Computes multi-head attention using a query, key and value.

    Args:
      q: Query with shape [batch, q_indices, num_heads, head_dim].
      k: Key with shape [batch, kv_indices, num_heads, head_dim].
      v: Value with shape [batch, kv_indices, num_heads, head_dim].
      dropout_prob: dropout probability on the attention weights.
      attention_mask: Array of shape [batch, q_indices, kv_indices] indicating
        which attentions are valid
    Returns:
      Output of the attention with shape [batch, q_indices, hiddens]
    """
    batch, q_indices, num_heads, q_head_dim = q.shape
    _, _, _, v_head_dim = v.shape
    hiddens = num_heads * v_head_dim

    attention = jnp.einsum('bthd,bThd->bhtT', q, k)

    scale = 1. / math.sqrt(q_head_dim)
    attention *= scale

    if attention_mask is not None:
        # Use large_k instead of np.NINF because np.NINF breaks for causal-masked
        # left-padded sampling.
        large_k = jnp.array(1e4 if attention.dtype == jnp.float16 else 1e30,
                            dtype=attention.dtype)

        attention = jnp.where(attention_mask[:, None, :, :], attention,
                              -large_k)

    scores = jax.nn.softmax(attention)
    if dropout_prob > 0:
        scores = hk.dropout(hk.next_rng_key(), dropout_prob, scores)
    summed = jnp.einsum('bhtT,bThd->bthd', scores, v)
    summed = jnp.reshape(summed, [batch, q_indices, hiddens])

    if attention_mask is not None:
        # If all attended tokens are masked, or for masked tokens
        # some rows of logits gets completely masked, in which case the softmax
        # gives a uniform row and we obtain non-zero outputs where it should be
        # zero. We force zeros.
        wipe_attn = jnp.all(
            attention_mask == 0, axis=2, keepdims=True)  # shape (B, T, 1)
        summed = jnp.where(wipe_attn, jnp.zeros_like(summed), summed)
    return summed, scores


def conv_1d(
        output_channels,
        init_scale=1.0,
        with_bias=True,
        name=None):
    """A 1D convolution."""
    return hk.Linear(
        output_size=output_channels,
        with_bias=with_bias,
        w_init=hk.initializers.VarianceScaling(init_scale),
        name=name)


def layer_norm(x, name=None):
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                        name=name)(x)


def make_cross_attention_mask(query_mask, kv_mask):
    batch_size, query_len = query_mask.shape
    _, key_len = kv_mask.shape
    mask = jax.vmap(jnp.outer)(query_mask, kv_mask)
    assert mask.shape == (batch_size, query_len, key_len)
    return mask


#  -----------------------------------------------------------
#  -----------------------  Modules  -------------------------
#  -----------------------------------------------------------


class Attention(hk.Module):
    """Multi-headed {cross, self}-attention."""

    def __init__(self,
                 num_heads=8,
                 init_scale=1.0,
                 with_final_bias=True,
                 final_init_scale_multiplier=1.,
                 dropout_prob=0.0,
                 qk_channels=None,
                 v_channels=None,
                 output_channels=None,
                 name=None):
        super(Attention, self).__init__(name=name)
        self._num_heads = num_heads
        self._init_scale = init_scale
        self._with_final_bias = with_final_bias
        self._final_init_scale = final_init_scale_multiplier * init_scale
        self._dropout_prob = dropout_prob

        # If none of these are passed, the Q input determines the output shape:
        self._qk_channels = qk_channels
        self._v_channels = v_channels
        self._output_channels = output_channels

    def __call__(self, inputs_q, inputs_kv, attention_mask=None):
        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if self._qk_channels is None:
            self._qk_channels = inputs_q.shape[-1]
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query
        # operation.
        if self._v_channels is None:
            self._v_channels = self._qk_channels
        # Project the output of QKV attention to a desired number of channels.
        # Default to the same number as the output of the QKV attention
        # operation.
        if self._output_channels is None:
            self._output_channels = self._v_channels

        if self._qk_channels % self._num_heads != 0:
            raise ValueError(f'qk_channels ({self._qk_channels}) must be divisible by'
                             f' num_heads ({self._num_heads}).')
        if self._v_channels % self._num_heads != 0:
            raise ValueError(f'v_channels ({self._v_channels}) must be divisible by'
                             f' num_heads ({self._num_heads}).')
        qk_channels_per_head = self._qk_channels // self._num_heads
        v_channels_per_head = self._v_channels // self._num_heads

        # Project QKV to a common feature dimension.
        q = conv_1d(self._qk_channels, init_scale=self._init_scale)(inputs_q)
        k = conv_1d(self._qk_channels, init_scale=self._init_scale)(inputs_kv)
        v = conv_1d(self._v_channels, init_scale=self._init_scale)(inputs_kv)

        # Reshape channels for multi-head attention.
        batch, q_time, _ = q.shape
        _, kv_time, _ = k.shape
        q = jnp.reshape(
            q, [batch, q_time, self._num_heads, qk_channels_per_head])
        k = jnp.reshape(
            k, [batch, kv_time, self._num_heads, qk_channels_per_head])
        v = jnp.reshape(
            v, [batch, kv_time, self._num_heads, v_channels_per_head])

        result, scores = attend(q, k, v, dropout_prob=self._dropout_prob,
                        attention_mask=attention_mask)
        return conv_1d(
            self._output_channels,
            with_bias=self._with_final_bias,
            init_scale=self._final_init_scale)(result), scores


class MLP(hk.Module):
    """A Transformer-style dense module to follow attention."""

    def __init__(self,
                 widening_factor=4,
                 dropout_prob=0.0,
                 init_scale=1.,
                 name=None):
        super(MLP, self).__init__(name=name)
        self._widening_factor = widening_factor
        self._dropout_prob = dropout_prob
        self._init_scale = init_scale

    def __call__(self, x, *, is_training):
        dropout_prob = self._dropout_prob if is_training else 0.0
        output_channels = x.shape[-1]
        x = conv_1d(
            output_channels=self._widening_factor * output_channels,
            init_scale=self._init_scale)(x)
        x = jax.nn.gelu(x)
        x = conv_1d(
            output_channels=output_channels,
            init_scale=self._init_scale)(x)
        return hk.dropout(hk.next_rng_key(), dropout_prob, x)


class SelfAttention(hk.Module):
    """A self-attention module, including a dense block."""

    def __init__(self,
                 widening_factor=4,
                 dropout_prob=0.0,
                 dropout_attn_prob=0.0,
                 num_heads=8,
                 att_init_scale=1.0,
                 dense_init_scale=1.0,
                 qk_channels=None,
                 v_channels=None,
                 name=None):
        super(SelfAttention, self).__init__(name=name)
        self._widening_factor = widening_factor
        self._dropout_prob = dropout_prob
        self._dropout_attn_prob = dropout_attn_prob
        self._num_heads = num_heads
        self._att_init_scale = att_init_scale
        self._dense_init_scale = dense_init_scale
        self._qk_channels = qk_channels
        self._v_channels = v_channels

    def __call__(self,
                 inputs,
                 *,
                 attention_mask=None,
                 is_training):
        dropout_prob = self._dropout_prob if is_training else 0.0
        dropout_attn_prob = self._dropout_attn_prob if is_training else 0.0

        x = inputs
        qkv_inputs = layer_norm(inputs)
        results, scores = Attention(
            num_heads=self._num_heads,
            init_scale=self._att_init_scale,
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            dropout_prob=dropout_attn_prob)(qkv_inputs, qkv_inputs,
                                            attention_mask=attention_mask)
        results = hk.dropout(hk.next_rng_key(), dropout_prob, results)
        x += results

        x += MLP(
            widening_factor=self._widening_factor,
            dropout_prob=dropout_prob,
            init_scale=self._dense_init_scale)(
                layer_norm(x), is_training=is_training)
        return x, scores


class CrossAttention(hk.Module):
    """A cross-attention module, including a dense block."""

    def __init__(self,
                 widening_factor=1,
                 dropout_prob=0.0,
                 dropout_attn_prob=0.0,
                 num_heads=8,
                 att_init_scale=1.0,
                 dense_init_scale=1.0,
                 shape_for_attn='kv',
                 use_query_residual=True,
                 qk_channels=None,
                 v_channels=None,
                 name=None):
        super(CrossAttention, self).__init__(name=name)
        self._widening_factor = widening_factor
        self._dropout_prob = dropout_prob
        self._dropout_attn_prob = dropout_attn_prob
        self._num_heads = num_heads
        self._att_init_scale = att_init_scale
        self._dense_init_scale = dense_init_scale
        self._shape_for_attn = shape_for_attn
        self._use_query_residual = use_query_residual
        self._qk_channels = qk_channels
        self._v_channels = v_channels

    def __call__(self,
                 inputs_q,
                 inputs_kv,
                 *,
                 attention_mask=None,
                 is_training):
        dropout_prob = self._dropout_prob if is_training else 0.0
        dropout_attn_prob = self._dropout_attn_prob if is_training else 0.0

        output_channels = inputs_q.shape[-1]
        if self._shape_for_attn == 'q':
            qk_channels = inputs_q.shape[-1]
        elif self._shape_for_attn == 'kv':
            qk_channels = inputs_kv.shape[-1]
        else:
            raise ValueError(f'Unknown value {self._shape_for_attn} for '
                             'shape_for_attention.')

        v_channels = None
        if self._qk_channels is not None:
            qk_channels = self._qk_channels
        if self._v_channels is not None:
            v_channels = self._v_channels

        results, scores = Attention(
            num_heads=self._num_heads,
            init_scale=self._att_init_scale,
            dropout_prob=dropout_attn_prob,
            qk_channels=qk_channels,
            v_channels=v_channels,
            output_channels=output_channels)(layer_norm(inputs_q),
                                             layer_norm(inputs_kv),
                                             attention_mask=attention_mask)
        results = hk.dropout(hk.next_rng_key(), dropout_prob, results)

        # Optionally include a residual to the query.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self._use_query_residual:
            x = inputs_q + results
        else:
            x = results

        x += MLP(
            widening_factor=self._widening_factor,
            dropout_prob=dropout_prob,
            init_scale=self._dense_init_scale)(
                layer_norm(x), is_training=is_training)
        return x, scores


#  -----------------------------------------------------------
#  -----------------------  Perceiver  -----------------------
#  -----------------------------------------------------------


class PerceiverEncoder(hk.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder."""

    def __init__(
        self,
        # The encoder has a total of
        #   num_self_attends_per_block * num_blocks
        # self-attend layers. We share weights between blocks.
        num_self_attends_per_block=6,
        num_blocks=8,
        z_index_dim=512,
        num_z_channels=1024,
        qk_channels=None,
        v_channels=None,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        cross_attend_widening_factor=1,
        self_attend_widening_factor=1,
        dropout_prob=0.0,
        z_pos_enc_init_scale=0.02,
        cross_attention_shape_for_attn='kv',
        use_query_residual=True,
            name='perceiver_encoder'):
        super().__init__(name=name)

        # Check that we can use multihead-attention with these shapes.
        if num_z_channels % num_self_attend_heads != 0:
            raise ValueError(f'num_z_channels ({num_z_channels}) must be divisible by'
                             f' num_self_attend_heads ({num_self_attend_heads}).')
        if num_z_channels % num_cross_attend_heads != 0:
            raise ValueError(f'num_z_channels ({num_z_channels}) must be divisible by'
                             f' num_cross_attend_heads ({num_cross_attend_heads}).')

        self._input_is_1d = True

        self._num_blocks = num_blocks

        # Construct the latent array initial state.
        self.z_pos_enc = position_encoding.TrainablePositionEncoding(
            index_dim=z_index_dim,
            num_channels=num_z_channels,
            init_scale=z_pos_enc_init_scale)

        # Construct the cross attend:
        self.cross_attend = CrossAttention(
            dropout_prob=dropout_prob,
            num_heads=num_cross_attend_heads,
            widening_factor=cross_attend_widening_factor,
            shape_for_attn=cross_attention_shape_for_attn,
            qk_channels=qk_channels,
            v_channels=v_channels,
            use_query_residual=use_query_residual)

        # Construct the block of self-attend layers.
        # We get deeper architectures by applying this block more than once.
        self.self_attends = []
        for _ in range(num_self_attends_per_block):
            self_attend = SelfAttention(
                num_heads=num_self_attend_heads,
                dropout_prob=dropout_prob,
                qk_channels=qk_channels,
                v_channels=v_channels,
                widening_factor=self_attend_widening_factor)
            self.self_attends.append(self_attend)

    def latents(self, inputs):
        # Initialize the latent array for the initial cross-attend.
        return self.z_pos_enc(batch_size=inputs.shape[0])

    def __call__(self, inputs, z, *, is_training, input_mask=None):
        attention_mask = None
        if input_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=jnp.ones(z.shape[:2], dtype=jnp.int32),
                kv_mask=input_mask)
        # Relevant: https://github.com/deepmind/deepmind-research/issues/253
        z, cross_scores = self.cross_attend(z, inputs, is_training=is_training,
                              attention_mask=attention_mask)

        all_self_scores = []
        for _ in range(self._num_blocks):
            for self_attend in self.self_attends:
                z, self_scores = self_attend(z, is_training=is_training)
                all_self_scores += [self_scores]
                # TODO: Return scores.
                # Rollout through xattn, _num_blocks attn?
        all_self_scores = jnp.array(all_self_scores)

        return z, (cross_scores, all_self_scores)


class AbstractPerceiverDecoder(hk.Module, metaclass=abc.ABCMeta):
    """Abstract Perceiver decoder."""

    @abc.abstractmethod
    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None,
                      subsampled_points=None):
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape(self, inputs):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, query, z, *, is_training, query_mask=None):
        raise NotImplementedError


class BasicDecoder(AbstractPerceiverDecoder):
    """Cross-attention-based decoder."""

    def __init__(self,
                 output_num_channels,
                 position_encoding_type='fourier',
                 # Ignored if position_encoding_type == 'none':
                 output_index_dims=None,
                 subsampled_index_dims=None,
                 num_z_channels=1024,
                 qk_channels=None,
                 v_channels=None,
                 use_query_residual=False,
                 output_w_init=None,
                 concat_preprocessed_input=False,
                 num_heads=1,
                 name='basic_decoder',
                 final_project=True,
                 **position_encoding_kwargs):
        super().__init__(name=name)
        self._position_encoding_type = position_encoding_type

        # If `none`, the decoder will not construct any position encodings.
        # You should construct your own when quering the decoder.
        self.output_pos_enc = None
        if self._position_encoding_type != 'none':
            self.output_pos_enc = position_encoding.build_position_encoding(
                position_encoding_type,
                index_dims=output_index_dims,
                **position_encoding_kwargs)

        self._output_index_dim = output_index_dims
        if subsampled_index_dims is None:
            subsampled_index_dims = output_index_dims
        self._subsampled_index_dims = subsampled_index_dims
        self._output_num_channels = output_num_channels
        self._output_w_init = output_w_init
        self._use_query_residual = use_query_residual
        self._qk_channels = qk_channels
        self._v_channels = v_channels
        self._final_project = final_project
        self._num_heads = num_heads

        self._concat_preprocessed_input = concat_preprocessed_input

    def output_shape(self, inputs):
        return ((inputs[0], self._subsampled_index_dims, self._output_num_channels),
                None)

    def decoder_query(self, inputs, modality_sizes=None,
                      inputs_without_pos=None, subsampled_points=None):
        assert self._position_encoding_type != 'none'  # Queries come from elsewhere
        if subsampled_points is not None:
            # unravel_index returns a tuple (x_idx, y_idx, ...)
            # stack to get the [n, d] tensor of coordinates
            pos = jnp.stack(
                jnp.unravel_index(subsampled_points, self._output_index_dim),
                axis=1)
            # Map these coordinates to [-1, 1]
            pos = -1 + 2 * pos / jnp.array(self._output_index_dim)[None, :]
            pos = jnp.broadcast_to(pos[None],
                                   [inputs.shape[0], pos.shape[0], pos.shape[1]])
            pos_emb = self.output_pos_enc(
                batch_size=inputs.shape[0],
                pos=pos)
            pos_emb = jnp.reshape(
                pos_emb, [pos_emb.shape[0], -1, pos_emb.shape[-1]])
        else:
            pos_emb = self.output_pos_enc(batch_size=inputs.shape[0])
        if self._concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError('Value is required for inputs_without_pos if'
                                 ' concat_preprocessed_input is True')
            pos_emb = jnp.concatenate([inputs_without_pos, pos_emb], axis=-1)

        return pos_emb

    def __call__(self, query, z, *, is_training,
                 query_mask=None):
        # Cross-attention decoding.
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        # Construct cross attention and linear layer lazily, in case we don't need
        # them.
        attention_mask = None
        if query_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=query_mask,
                kv_mask=jnp.ones(z.shape[:2], dtype=jnp.int32))
        decoding_cross_attn = CrossAttention(
            dropout_prob=0.0,
            num_heads=self._num_heads,
            widening_factor=1,
            shape_for_attn='kv',
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            use_query_residual=self._use_query_residual)
        output, scores = decoding_cross_attn(query, z, is_training=is_training,
                                     attention_mask=attention_mask)
        if self._final_project:
            final_layer = hk.Linear(
                self._output_num_channels, w_init=self._output_w_init, name='output')
            output = final_layer(output)
        return output
