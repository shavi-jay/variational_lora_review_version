import logging
from typing import Dict, Tuple, Optional

import torch
from torch import nn
from transformers import (
    BertPreTrainedModel,
    BertModel,
    BertConfig,
)
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MolbertBatchType = Tuple[
    Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], torch.Tensor
]


# combined with deprecated modeling_transfo_xl directory in transformers
class SuperPositionalEmbedding(nn.Module):
    """
    Same as PositionalEmbedding in XLTransformer, BUT
    has a different handling of the batch dimension that avoids cumbersome dimension shuffling
    """

    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        pos_emb = pos_emb.unsqueeze(0)
        if bsz is not None:
            pos_emb = pos_emb.expand(bsz, -1, -1)
        return pos_emb


class SuperPositionalBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings, BUT
    uses non-learnt (computed) positional embeddings
    """

    def __init__(self, config):
        super(SuperPositionalBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = SuperPositionalEmbedding(config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # do word embedding first to determine its type (float or half)
        words_embeddings = self.word_embeddings(input_ids)

        # if position_ids or token_type_ids were not provided, used defaults
        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(
                seq_length, dtype=words_embeddings.dtype, device=words_embeddings.device
            )
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if inputs_embeds is None:
            inputs_embeds = words_embeddings
        position_embeddings = self.position_embeddings(position_ids, input_ids.size(0))
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SuperPositionalBertModel(BertModel):
    """
    Same as BertModel, BUT
    uses SuperPositionalBertEmbeddings instead of BertEmbeddings
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = SuperPositionalBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()
        self.attn_implementation = "eager"


class MolbertPretrainedModel(BertPreTrainedModel):
    """
    General BERT model with tasks to specify
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = SuperPositionalBertModel(config)
        self.bert.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        output = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )

        return output

    def load_from_pretrained(
        self,
        load_path: str,
    ):
        data = torch.load(load_path)
        self.load_state_dict(data)


class BertConfigExtras(BertConfig):
    """
    Same as BertConfig, BUT
    adds any kwarg as a member field
    """

    def __init__(
        self,
        vocab_size=42,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        **kwargs,
    ):
        super(BertConfigExtras, self).__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
        )

        for k, v in kwargs.items():
            setattr(self, k, v)
