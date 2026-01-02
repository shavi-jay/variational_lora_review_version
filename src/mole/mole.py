import copy

import torch

from src.mole.deberta.cache_utils import load_model_state
from src.mole.bert import BertEmbeddings, BertEncoder


class AtomEnvEmbeddings(torch.nn.Module):
    """ AtomEnvEmbeddings is a DeBERTa encoder
  This module is composed of the input embedding layer with stacked transformer layers with disentangled attention.

  Parameters:
    config:
      A model config class instance with the configuration to build a new model. The schema is \
          similar to `BertConfig`, for more details, please refer :class:`~DeBERTa.deberta.ModelConfig`

    pre_trained:
      The pre-trained DeBERTa model, it can be a physical path of a pre-trained DeBERTa model or \
          a released configurations, i.e. [**base, large, base_mnli, large_mnli**]

  """

    def __init__(self, config=None, pre_trained=None):
        super().__init__()
        state = None
        if pre_trained is not None:
            state, model_config = load_model_state(pre_trained)
            if config is not None and model_config is not None:
                for k in config.__dict__:
                    if k not in [
                        "hidden_size",
                        "intermediate_size",
                        "num_attention_heads",
                        "num_hidden_layers",
                        "vocab_size",
                        "max_position_embeddings",
                    ]:
                        model_config.__dict__[k] = config.__dict__[k]
            config = copy.copy(model_config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.config = config
        self.pre_trained = pre_trained
        self.apply_state(state)

    def forward(
        self,
        input_ids,
        input_mask=None,
        attention_mask=None,
        token_type_ids=None,
        output_all_encoded_layers=True,
        position_ids=None,
        return_att=False,
        relative_pos=None,
    ):
        if input_mask is None:
            input_mask = torch.ones_like(input_ids)
        if attention_mask is None:
            attention_mask = input_mask
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        ebd_output = self.embeddings(
            input_ids.to(torch.long),
            token_type_ids.to(torch.long),
            position_ids,
            input_mask,
        )
        embedding_output = ebd_output["embeddings"]
        encoder_output = self.encoder(
            embedding_output,
            attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            return_att=return_att,
            relative_pos=relative_pos,
        )
        encoder_output.update(ebd_output)
        return encoder_output

    def apply_state(self, state=None):
        """ Load state from previous loaded model state dictionary.

      Args:
        state (:obj:`dict`, optional): State dictionary as the state returned by torch.module.state_dict(), \
            default: `None`. \
            If it's `None`, then will use the pre-trained state loaded via the constructor to re-initialize \
            the `DeBERTa` model
    """
        if self.pre_trained is None and state is None:
            return
        if state is None:
            state, config = load_model_state(self.pre_trained)
            self.config = config

        prefix = ""
        for k in state:
            if "embeddings." in k:
                if not k.startswith("embeddings."):
                    prefix = k[: k.index("embeddings.")]
                break

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        self._load_from_state_dict(
            state,
            prefix=prefix,
            local_metadata=None,
            strict=True,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
