import logging
import os
from abc import abstractmethod, ABC
from typing import List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MolFeaturizer(ABC):
    """
    Interface for the featurization of molecules, given as SMILES strings, to some vectorized representation.
    """

    def __call__(
        self, molecules: Sequence[str], **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.transform(molecules)

    def transform(self, molecules: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Featurizes a sequence of molecules.

        Args:
            molecules: molecules, given as a sequence of SMILES strings

        Returns:
            Tuple: 2D array for the feature vectors, 1D array for the validity masks
        """
        single_results = [self.transform_single(m) for m in molecules]
        features_list, mask_list = zip(*single_results)

        return {
            "input_ids": torch.tensor(np.vstack(features_list)),
            "valid": torch.tensor(np.hstack(mask_list)),
        }

    @abstractmethod
    def transform_single(self, molecule: str) -> Tuple[np.ndarray, bool]:
        """
        Featurizes one molecule.

        Args:
            molecule: molecule, given as a SMILES string

        Returns:
            Tuple: feature vector (1D array), boolean for successful featurization
        """

    def invalid_mol_features(self) -> np.ndarray:
        """
        Features to return for invalid molecules.
        """
        return np.zeros(self.output_size)

    @property
    @abstractmethod
    def output_size(self) -> int:
        """
        Get the dimension after featurization
        """

    def is_valid(self, molecules: Sequence[str]) -> Sequence[bool]:
        return np.array([self.is_valid_single(mol) for mol in molecules])

    def is_valid_single(self, molecule: str) -> bool:
        mol = Chem.MolFromSmiles(molecule, True, {})

        if mol is None or len(molecule) == 0:
            return False

        return True


class SmilesIndexFeaturizer(MolFeaturizer):
    """
    Transforms a SMILES string into its index character representation
    Each double letter element is first converted into a single symbol
    """

    def __init__(
        self,
        max_length: int,
        pad: str = "☐",
        begin: str = "^",
        end: str = "$",
        allowed_elements: tuple = (
            "F",
            "H",
            "I",
            "B",
            "C",
            "N",
            "O",
            "P",
            "S",
            "Br",
            "Cl",
            "Si",
            "Se",
            "se",
            "@@",
        ),
        extra_symbols: Optional[List[str]] = None,
        canonicalise: bool = True,
        permute: bool = False,
    ) -> None:

        self.max_length = max_length
        self.pad = pad
        self.begin = begin
        self.end = end
        self.allowed_elements = allowed_elements
        self.extra_symbols = [] if extra_symbols is None else extra_symbols
        self.symbols = [s for s in [self.pad, self.begin, self.end] if s is not None]
        self.symbols += self.extra_symbols
        self.canonicalise = canonicalise
        self.permute = permute

        assert not (
            self.permute and self.canonicalise
        ), "Cannot have both permute and canonicalise equal True"

        assert pad is not None, "PAD symbol cannot be None!"
        assert pad != begin and pad != end
        assert begin != end or (begin is None and end is None)

        self.elements, self.chars = self.load_periodic_table()

        self.forbidden_symbols = set(self.elements) - set(allowed_elements)

        self.encode_dict = {
            symbol: char
            for symbol, char in zip(self.elements, self.chars)
            if symbol in self.allowed_elements and len(symbol) > 1
        }

        self.decode_dict = {v: k for k, v in self.encode_dict.items()}

        self.allowed_elements_chars = [
            e if len(e) == 1 else self.encode_dict[e] for e in self.allowed_elements
        ]

        self.smiles_special_chars = (
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "=",
            "@",
            "#",
            "%",
            "/",
            "\\",
            "(",
            ")",
            "+",
            "-",
            ".",
            "[",
            "]",
        )

        self.idx_to_token = [
            *self.symbols,
            *self.allowed_elements_chars,
            *self.smiles_special_chars,
        ]

        self.token_to_idx = {v: k for k, v in enumerate(self.idx_to_token)}

    @staticmethod
    def load_periodic_table() -> Tuple[List[str], List[str]]:
        this_directory = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(this_directory, "elements.txt")
        df = pd.read_csv(data_path)
        names = df["symbol"].to_list()
        chars = df["char"].to_list()
        return names, chars

    def is_legal(self, smiles: str) -> bool:
        """
        Determine if smiles string has illegal symbols

        Args:
            smiles: SMILES string

        Returns:
            True if all legal
        """
        for symbol in self.forbidden_symbols:
            if symbol in smiles:
                logging.warning(f"SMILES has forbidden symbol! {smiles} -> {symbol}")
                return False
        return True

    def is_short(self, smiles: List[str]) -> bool:
        """
        Determine if input string is not too long
        It should be already standardised and encoded

        Args:
            smiles: SMILES string

        Returns:
            True if not too long
        """
        short_enough = (
            len(smiles) <= self.max_length if self.max_length is not None else True
        )
        if not short_enough:
            logging.warning(f"SMILES is too long! {smiles}")
        return short_enough

    def standardise(
        self, smiles: str, canonicalise: Optional[bool] = None
    ) -> Optional[str]:
        """
        Standardise a SMILES string if valid (canonical + kekulized)

        Args:
            smiles: SMILES string
            canonicalise: optional flag to override `self.canonicalise`

        Returns: standard version the SMILES if valid, None otherwise

        """
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
        except Exception as e:
            # invalid?
            logging.warning(f'Chem.MolFromSmiles failed smiles="{smiles}" error={e}')
            return None

        if mol is None:
            # invalid?
            logging.warning(f'Chem.MolFromSmiles failed smiles="{smiles}"')
            return None

        flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP
        Chem.SanitizeMol(mol, flags, catchErrors=True)

        if self.canonicalise or canonicalise:
            # bug where permuted smiles are not canonicalised to the same form. This is fixed by round tripping SMILES
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            if mol is None:
                logging.warning(
                    f'Chem.MolFromSmiles failed after sanitization smiles="{smiles}"'
                )
                return None

        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
            smiles = Chem.MolToSmiles(
                mol, kekuleSmiles=True, canonical=self.canonicalise or canonicalise
            )
        except (ValueError, RuntimeError):
            logging.warning(f"SMILES failed Kekulization! {smiles}")
            return None

        return smiles

    def encode(self, smiles: str) -> str:
        """
        Replace multi-char tokens with single tokens in SMILES string.

        Args:
            smiles: SMILES string
        Returns:
            sanitized SMILE string with only single-char tokens
        """

        temp_smiles = smiles
        for symbol, token in self.encode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        return temp_smiles

    def decode(self, smiles: str) -> str:
        """
        Replace special tokens with their multi-character equivalents.

        Args:
            smiles: SMILES string

        Returns:
            SMILES string with possibly multi-char tokens
        """
        temp_smiles = smiles
        for symbol, token in self.decode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        return temp_smiles

    def decorate(self, smiles: List[str]) -> List[str]:
        """
        Add optional BEGIN and END symbols if available

        Args:
            smiles: SMILES string

        Returns:
            decorated SMILES string
        """
        if self.begin is not None:
            smiles = [self.begin] + smiles
        if self.end is not None:
            smiles = smiles + [self.end]
        return smiles

    @property
    def vocab_size(self) -> int:
        """
        Number of available symbols
        """
        return len(self.idx_to_token)

    @property
    def begin_idx(self) -> Optional[int]:
        return self.token_to_idx.get(self.begin)

    @property
    def end_idx(self) -> Optional[int]:
        return self.token_to_idx.get(self.end)

    @property
    def pad_idx(self) -> Optional[int]:
        return self.token_to_idx.get(self.pad)

    @property
    def output_size(self):
        return self.max_length

    def matrix_to_smiles(self, array: np.ndarray, trim: bool = False) -> List[str]:
        """
        Converts an matrix of indices into their SMILES representations

        Args:
            array: torch tensor of indices, one molecule per row
            trim: remove special characters

        Returns:
            list of SMILES, without the termination symbol
        """
        smiles_strings = []

        for row in array:

            predicted_chars = []

            for j in row:
                next_char = self.idx_to_token[j.item()]
                predicted_chars.append(next_char)

            smi = "".join(predicted_chars)
            smi = self.decode(smi)

            if trim:
                if self.pad:
                    smi = smi.replace(self.pad, "")
                if self.begin:
                    smi = smi.replace(self.begin, "")
                if self.end:
                    smi = smi.replace(self.end, "")

            smiles_strings.append(smi)

        return smiles_strings

    def transform_single(self, molecule: str) -> Tuple[np.ndarray, bool]:
        """
        Transform a single amino acid sequence

        Args:
            molecule: SMILES string

        Returns:
            single character index representation, valid mask

        Issues:

         The extra return on standardize is below

         >>> from rdkit import Chem, RDLogger
         ... smiles = 'c1(cc(N\C(=[NH]\c2cccc(c2)CC)C)ccc1)CC'
         ... mol = Chem.MolFromSmiles(smiles, sanitize=False)
         ... flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP
         ... Chem.SanitizeMol(mol, flags, catchErrors=True)

         Will give valid mol that cant be standardized!

        """
        indices_array = np.full(self.max_length, fill_value=self.pad_idx)

        if not self.is_valid_single(molecule):
            return indices_array, False

        # check that encode hasn't been called already (alchemy bugfix 1197)
        for symbol in self.encode_dict.values():
            if symbol in molecule:
                logging.warning(
                    f"SMILES has already been encoded, contains {symbol}: {molecule}"
                )
                return indices_array, False

        if self.permute:
            standard_smiles = self.permute_smiles(molecule)
        else:
            standard_smiles = self.standardise(molecule)

        if standard_smiles is None:
            return indices_array, False

        single_char_smiles = self.encode(standard_smiles)
        decorated_smiles = self.decorate(list(single_char_smiles))
        valid_smiles = self.is_legal(standard_smiles) and self.is_short(
            decorated_smiles
        )

        if valid_smiles:
            for i, c in enumerate(decorated_smiles):
                try:
                    indices_array[i] = self.token_to_idx[c]
                except KeyError:
                    logging.warning(
                        f"SMILES has unknown symbol {decorated_smiles} -> {c}"
                    )

        return indices_array, valid_smiles

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> List[int]:
        """Converts a sequence of tokens into ids using the vocab."""

        ids = [self.token_to_idx[token] for token in tokens]

        if len(ids) > self.max_length:
            logging.warning(
                f"Token indices sequence length is longer than the specified maximum "
                f"sequence length for this BERT model ({len(ids)} > {self.max_length}). "
                f"Running this sequence through BERT will result in indexing errors"
            )
        return ids

    def permute_smiles(self, smiles_str: str, seed: int = None) -> Optional[str]:
        """
        Permute the input smiles.

        Args:
          smiles_str: The smiles input

        Returns:
          The standardised permuted smiles.
        """
        if seed is not None:
            np.random.seed(seed)

        try:
            mol = Chem.MolFromSmiles(smiles_str, sanitize=False)
        except Exception as e:
            logging.warning(
                f'Chem.MolFromSmiles failed smiles="{smiles_str}" error={e}'
            )
            return None

        if mol is None:
            # invalid?
            logging.warning(f'Chem.MolFromSmiles failed smiles="{smiles_str}"')
            return None

        # get atom list and shuffle
        ans = list(range(mol.GetNumAtoms()))
        np.random.shuffle(ans)

        # re-order the molecule
        smiles = Chem.MolToSmiles(Chem.RenumberAtoms(mol, ans), canonical=False)

        # standardise and return
        return self.standardise(smiles)

    @classmethod
    def bert_smiles_index_featurizer(
        cls,
        max_length: int,
        allowed_elements: tuple = (
            "F",
            "H",
            "I",
            "B",
            "C",
            "N",
            "O",
            "P",
            "S",
            "Br",
            "Cl",
            "Si",
            "Se",
            "se",
            "@@",
        ),
        canonicalise: bool = False,
        permute: bool = False,
    ):
        """
        Bert specific index featurizer
        """
        return cls(
            max_length=max_length,
            pad="[PAD]",
            begin="[CLS]",
            end="[SEP]",
            allowed_elements=allowed_elements,
            extra_symbols=["[MASK]"],
            canonicalise=canonicalise,
            permute=permute,
        )
