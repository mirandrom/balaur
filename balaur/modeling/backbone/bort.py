from dataclasses import dataclass
import logging
import math

import torch
from torch import nn
from balaur.modeling.utils.activations import ACT2FN

logger = logging.getLogger(__name__)


@dataclass
class BortConfig:
    model_seed: int = 42
    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    activation: str = "gelu"
    embed_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    mask_token_id: int = 50264
    encoder_ln_mode: str = "pre_ln"
    embed_layernorm: bool = False
    mlm_bias: bool = False
    sparse_pred: bool = True


class BortEmbeddings(nn.Module):
    def __init__(self, c: BortConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(c.vocab_size, c.hidden_size)
        self.position_embeddings = nn.Embedding(c.max_position_embeddings, c.hidden_size)
        self.dropout = nn.Dropout(c.embed_dropout_prob)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class BortAttention(nn.Module):
    def __init__(self, c: BortConfig):
        super().__init__()
        assert c.hidden_size % c.num_attention_heads == 0
        self.num_attention_heads = c.num_attention_heads
        self.attention_head_size = int(c.hidden_size / c.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(c.hidden_size, self.all_head_size)
        self.key = nn.Linear(c.hidden_size, self.all_head_size)
        self.value = nn.Linear(c.hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.dense = nn.Linear(c.hidden_size, c.hidden_size)
        self.attention_dropout = nn.Dropout(c.attention_dropout_prob)
        self.hidden_dropout = nn.Dropout(c.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        # (bsz, seq, nheads*head_dim) -> (bsz, nheads, seq, head_dim)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        # (bsz, seq, nheads*head_dim) -> (bsz, nheads, head_dim, seq)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 3, 1)

    def transpose_contextual(self, x):
        # (bsz, nheads, seq, head_dim) -> (bsz, seq, nheads*head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (self.all_head_size,)
        return x.view(*new_x_shape)

    def forward(self, hidden_states, attention_mask):
        # QKV matrices, concatenated across heads (bsz, seq, nheads*head_dim)
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # QKV matrices, split across heads
        query_layer = self.transpose_for_scores(query_layer)    # (bsz, nheads, seq, head_dim)
        value_layer = self.transpose_for_scores(value_layer)    # (bsz, nheads, seq, head_dim)
        key_layer = self.transpose_key_for_scores(key_layer)    # (bsz, nheads, head_dim, seq)

        # Attention scores (bsz, nheads, seq, seq)
        x = torch.matmul(query_layer, key_layer)
        x = x / math.sqrt(self.attention_head_size)
        x = x + attention_mask
        x = self.softmax(x)
        x = self.attention_dropout(x)

        # Contextual embeddings
        x = torch.matmul(x, value_layer)       # (bsz, nheads, seq, head_dim)
        x = self.transpose_contextual(x)       # (bsz, seq, nheads*head_dim)
        x = self.dense(x)                      # (bsz, seq, dim)
        x = self.hidden_dropout(x)
        return x


class BortFFN(nn.Module):
    def __init__(self, c: BortConfig):
        super().__init__()
        self.dense_act = nn.Linear(c.hidden_size, c.intermediate_size)
        self.activation = ACT2FN[c.activation]
        self.dense = nn.Linear(c.intermediate_size, c.hidden_size)
        self.hidden_dropout = nn.Dropout(c.hidden_dropout_prob)

    def forward(self, x):
        x = self.activation(self.dense_act(x))
        x = self.dense(x)
        x = self.hidden_dropout(x)
        return x


class BortLayer(nn.Module):
    def __init__(self, c: BortConfig):
        super().__init__()
        self.attention = BortAttention(c)
        self.encoder_ln_mode = c.encoder_ln_mode
        self.LayerNorm1 = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.LayerNorm2 = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.ffn = BortFFN(c)

    def pre_ln_forward(self, x, attention_mask):
        x1 = self.LayerNorm1(x)
        x2 = x + self.attention(x1, attention_mask)
        x3 = self.LayerNorm2(x2)
        x4 = x2 + self.ffn(x3)
        return x4

    def post_ln_forward(self, x, attention_mask):
        x += self.attention(x, attention_mask)
        x = self.LayerNorm1(x)
        x += self.ffn(x)
        x = self.LayerNorm2(x)
        return x

    def forward(self, hidden_states, attention_mask):
        if self.encoder_ln_mode == 'pre_ln':
            return self.pre_ln_forward(hidden_states, attention_mask)
        elif self.encoder_ln_mode == 'post_ln':
            return self.post_ln_forward(hidden_states, attention_mask)
        else:
            raise ValueError(f"Invalid encoder_ln_mode: {self.encoder_ln_mode}")


class BortEncoder(nn.Module):
    def __init__(self, c: BortConfig):
        super().__init__()
        self.layer = nn.ModuleList([BortLayer(c) for _ in range(c.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BortBackbone(nn.Module):
    def __init__(self, c: BortConfig):
        super().__init__()
        self.embeddings = BortEmbeddings(c)
        self.pad_token_id = c.pad_token_id
        self.encoder = BortEncoder(c)
        self.ln_encoder_output = c.encoder_ln_mode == "pre_ln"
        if self.ln_encoder_output:
            self.LayerNorm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)

    def prepare_attention_mask(self, attention_mask):
        # (bsz,1,1,seq) to broadcast with attention logits (bsz,num_heads,seq,seq)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # ensure dtype is compatible with rest of model for fp16 training
        attention_mask = attention_mask.to(dtype=self.embeddings.word_embeddings.weight.dtype)
        # this mask is added to pre-softmax logits, so a highly negative value zeros probability
        attention_mask = (1.0 - attention_mask) * -10000.0
        return attention_mask

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = self.prepare_attention_mask(attention_mask)
        embedding_output = self.embeddings(input_ids)
        encoder_output = self.encoder(embedding_output, attention_mask)
        if self.ln_encoder_output:
            encoder_output = self.LayerNorm(encoder_output)
        return encoder_output


class BortMLMHead(nn.Module):
    def __init__(self, c: BortConfig):
        super().__init__()
        self.dense_act = nn.Linear(c.hidden_size, c.hidden_size)
        self.activation = ACT2FN[c.activation]
        self.LayerNorm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.decoder = nn.Linear(c.hidden_size, c.vocab_size, bias=c.mlm_bias)

    def forward(self, x):
        x = self.activation(self.dense_act(x))
        x = self.LayerNorm(x)
        x = self.decoder(x)
        return x


class BortForMLM(nn.Module):
    def __init__(self, config: BortConfig):
        super().__init__()
        self.config = config
        self.bort = BortBackbone(config)
        self.mlm_head = BortMLMHead(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)
        self.mlm_head.decoder.weight = self.bort.embeddings.word_embeddings.weight

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=True):
        hidden_states = self.bort(input_ids, attention_mask=attention_mask)
        mask_unmasked = input_ids.view(-1) == self.config.mask_token_id
        x = hidden_states.view(-1, hidden_states.shape[-1])[mask_unmasked]
        x = self.mlm_head(x)
        loss = None
        if labels is not None:
            labels = labels.view(-1)[mask_unmasked]
            loss = self.loss_fct(x, labels)

        return dict(
            loss=loss,
            logits=x,
            hidden_states=hidden_states if output_hidden_states else None
        )


class BortForSequenceClassification(nn.Module):
    def __init__(self,
                 config: BortConfig,
                 num_labels: int = 2,
                 pooler_activation: str = "tanh"):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.bort = BortBackbone(config)

        self.pooler_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = ACT2FN[pooler_activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.bort(input_ids, attention_mask=attention_mask)
        pooled_output = self.pooler_activation(self.pooler_linear(hidden_states[:, 0]))
        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return dict(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )
