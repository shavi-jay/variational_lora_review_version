from src.mole.datasets import MOLE_VOCAB_PATH, open_dictionary, getAtomEnvironments
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj


class MolETokenizer:
    def __init__(self, vocab_path=MOLE_VOCAB_PATH, radius_inp=0, cls_token=True):
        self.dictionary = open_dictionary(dictionary_path=vocab_path)
        self.inv_dictionary = {v: k for k, v in self.dictionary.items()}
        self.radius_inp = radius_inp
        self.cls_token = cls_token

        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius_inp,
            includeRedundantEnvironments=True,
        )

    def tokenize(self, smiles_list: list[str], *args, **kwargs):
        data_list = []
        for smiles in smiles_list:
            data = self._smiles_to_data(smiles)
            data_list.append(data)
        batch = Batch.from_data_list(data_list)

        return self._batch_to_inputs(batch)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        raise NotImplementedError()

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        raise NotImplementedError()

    def __call__(self, *args, **kwds):
        return self.tokenize(*args, **kwds)

    def _smiles_to_data(self, smiles: str):
        data_dict = {}
        mol = Chem.MolFromSmiles(smiles)
        dist_mat = Chem.GetDistanceMatrix(mol)
        # atomenv_inp = getAtomEnvironments(
        #     mol, self.dictionary_inp, self.radius_inp, self.useFeatures_inp
        # )
        atomenv_inp = getAtomEnvironments(
            mol, self.dictionary, self.radius_inp, self.mfpgen
        )

        # Get unmasked input tokens and labels for supervised training
        tokens = atomenv_inp
        data_dict.update(
            {
                "x": torch.tensor(tokens, dtype=torch.long),
            }
        )

        # Transform distance matrix into a sparse matrix
        dist_mat[dist_mat == 1.0e08] = -1
        dist_mat = sparse.coo_matrix(dist_mat + 1)

        if self.cls_token:
            # Add CLS token at position '0' for input tokens, lables and distance matrix
            tokens = np.insert(tokens, 0, self.dictionary["CLS"])
            data_dict.update(
                {
                    "x": torch.tensor(tokens, dtype=torch.long),
                }
            )

            dist_mat = sparse.vstack(
                ((np.zeros(dist_mat.shape[0])[None, :]), dist_mat)
            )  # use 0 as padding index
            dist_mat = sparse.hstack(
                ((np.zeros(dist_mat.shape[0])[:, None]), dist_mat)
            )  # use 0 as padding index

        data_dict.update(
            {
                "edge_index": torch.tensor(
                    np.array([dist_mat.row, dist_mat.col]), dtype=torch.long
                ),
                "edge_attr": torch.tensor(dist_mat.data, dtype=torch.long),
            }
        )
        return Data.from_dict(data_dict)

    def _batch_to_inputs(self, batch: Batch):
        input_ids, input_mask = to_dense_batch(batch.x, batch.batch, fill_value=0)
        relative_pos = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)

        return {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "relative_pos": relative_pos,
        }
