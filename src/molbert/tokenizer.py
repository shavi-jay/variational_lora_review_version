from src.molbert.featurizer import SmilesIndexFeaturizer
from transformers import BertTokenizer
import os

MOLBERT_VOCAB_PATH = "molbert_vocab.txt"


class MolbertTokenizer(BertTokenizer):
    def __init__(
        self,
        max_sequence_length=512,
        vocab_file=None,
        do_lower_case=False,
        unk_token="<pad>",
        sep_token="<eos>",
        pad_token="<pad>",
        cls_token="<bos>",
        mask_token="<mask>",
        **kwargs
    ):
        if vocab_file is None:
            this_directory = os.path.dirname(os.path.realpath(__file__))
            vocab_file = os.path.join(this_directory, MOLBERT_VOCAB_PATH)
        super().__init__(
            vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )

        self.smiles_featurizer = SmilesIndexFeaturizer(
            max_sequence_length, pad="<pad>", begin="<bos>", end="<eos>"
        )
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None

    def _tokenize(self, text):
        if not self.smiles_featurizer.is_valid_single(text):
            return []

        if self.smiles_featurizer.permute:
            smiles = self.smiles_featurizer.permute_smiles(text)
        else:
            smiles = self.smiles_featurizer.standardise(text)

        if smiles is None:
            return []

        tokens = self.smiles_featurizer.encode(smiles)

        return tokens

    def convert_tokens_to_string(self, tokens):
        decoded = []

        for token in tokens:
            if token in self.smiles_featurizer.decode_dict:
                decoded.append(self.smiles_featurizer.decode_dict[token])
            else:
                decoded.append(token)

        out_string = "".join(decoded)
        return out_string
