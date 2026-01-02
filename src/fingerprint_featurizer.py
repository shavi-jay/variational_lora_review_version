"""
Topological fingerprints.
"""

from typing import Dict, List, Callable, Union, Sequence

import numpy as np
from scipy.stats import rv_continuous

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.rdkit_utils import DescriptorsNormalizationParameters as DNP

from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MolToSmiles

import logging

logger = logging.getLogger(__name__)


class CircularFingerprint(MolecularFeaturizer):
    """Circular (Morgan) fingerprints.

    Extended Connectivity Circular Fingerprints compute a bag-of-words style
    representation of a molecule by breaking it into local neighborhoods and
    hashing into a bit vector of the specified size. It is used specifically
    for structure-activity modelling. See [1]_ for more details.

    References
    ----------
    .. [1] Rogers, David, and Mathew Hahn. "Extended-connectivity fingerprints."
        Journal of chemical information and modeling 50.5 (2010): 742-754.

    Note
    ----
    This class requires RDKit to be installed.

    Examples
    --------
    >>> import deepchem as dc
    >>> from rdkit import Chem
    >>> smiles = ['C1=CC=CC=C1']
    >>> # Example 1: (size = 2048, radius = 4)
    >>> featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
    >>> features = featurizer.featurize(smiles)
    >>> type(features[0])
    <class 'numpy.ndarray'>
    >>> features[0].shape
    (2048,)

    >>> # Example 2: (size = 2048, radius = 4, sparse = True, smiles = True)
    >>> featurizer = dc.feat.CircularFingerprint(size=2048, radius=8,
    ...                                          sparse=True, smiles=True)
    >>> features = featurizer.featurize(smiles)
    >>> type(features[0]) # dict containing fingerprints
    <class 'dict'>

    """

    def __init__(
        self,
        radius: int = 2,
        size: int = 2048,
        chiral: bool = False,
        bonds: bool = True,
        features: bool = False,
        sparse: bool = False,
        smiles: bool = False,
        is_counts_based: bool = False,
    ):
        """
        Parameters
        ----------
        radius: int, optional (default 2)
            Fingerprint radius.
        size: int, optional (default 2048)
            Length of generated bit vector.
        chiral: bool, optional (default False)
            Whether to consider chirality in fingerprint generation.
        bonds: bool, optional (default True)
            Whether to consider bond order in fingerprint generation.
        features: bool, optional (default False)
            Whether to use feature information instead of atom information; see
            RDKit docs for more info.
        sparse: bool, optional (default False)
            Whether to return a dict for each molecule containing the sparse
            fingerprint.
        smiles: bool, optional (default False)
            Whether to calculate SMILES strings for fragment IDs (only applicable
            when calculating sparse fingerprints).
        is_counts_based: bool, optional (default False)
            Whether to generates a counts-based fingerprint.

        """
        self.radius = radius
        self.size = size
        self.chiral = chiral
        self.bonds = bonds
        self.features = features
        self.sparse = sparse
        self.smiles = smiles
        self.is_counts_based = is_counts_based
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.size,
            includeChirality=self.chiral,
            useBondTypes=self.bonds,
            onlyNonzeroInvariants=self.features,
        )

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
        """Calculate circular fingerprint.

        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
            RDKit Mol object

        Returns
        -------
        np.ndarray
            A numpy array of circular fingerprint.

        """
        try:
            from rdkit import Chem, DataStructs
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")
        if "mol" in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )
        if self.sparse:
            info: Dict = {}
            fp = self.mfpgen.GetFingerprint(
                datapoint,
            )
            fp = fp.GetNonzeroElements()  # convert to a dict

            # generate SMILES for fragments
            if self.smiles:
                fp_smiles = {}
                for fragment_id, count in fp.items():
                    root, radius = info[fragment_id][0]
                    env = Chem.FindAtomEnvironmentOfRadiusN(datapoint, radius, root)
                    frag = Chem.PathToSubmol(datapoint, env)
                    smiles = Chem.MolToSmiles(frag)
                    fp_smiles[fragment_id] = {"smiles": smiles, "count": count}
                fp = fp_smiles
        else:
            if self.is_counts_based:
                fp_sparse = self.mfpgen.GetCountFingerprint(
                    datapoint,
                )
                fp = np.zeros(
                    (self.size,), dtype=float
                )  # initialise numpy array of zeros (shape: (required size,))
                DataStructs.ConvertToNumpyArray(fp_sparse, fp)
            else:
                # fp = self.mfpgen.GetCountFingerprintAsNumPy(
                #     datapoint,
                # )
                fp_sparse = self.mfpgen.GetFingerprint(
                    datapoint,
                )
                fp = np.zeros(
                    (self.size,), dtype=float
                )  # initialise numpy array of zeros (shape: (required size,))
                DataStructs.ConvertToNumpyArray(fp_sparse, fp)
        return fp

    def __hash__(self):
        return hash(
            (
                self.radius,
                self.size,
                self.chiral,
                self.bonds,
                self.features,
                self.sparse,
                self.smiles,
            )
        )

    def __eq__(self, other):
        if not isinstance(self, other.__class__):
            return False
        return (
            self.radius == other.radius
            and self.size == other.size
            and self.chiral == other.chiral
            and self.bonds == other.bonds
            and self.features == other.features
            and self.sparse == other.sparse
            and self.smiles == other.smiles
        )


######
# Combined RDKit descriptors and ECFP fingerprints
######


class ConcatenatedRDKitDescriptorsCircularFingerprint(MolecularFeaturizer):
    """
    RDKit descriptors and Circular Fingerprint concatenation.

    Note see `RDKitDescriptors` and `CircularFingerprint` for further details.
    """

    def __init__(
        self,
        descriptors: List[str] = [],
        is_normalized: bool = False,
        use_fragment: bool = True,
        ipc_avg: bool = True,
        use_bcut2d: bool = True,
        labels_only: bool = False,
        radius: int = 2,
        size: int = 2048,
        chiral: bool = False,
        bonds: bool = True,
        features: bool = False,
        sparse: bool = False,
        smiles: bool = False,
        is_counts_based: bool = False,
    ):
        """Initialize this featurizer.

        Parameters
        ----------
        RDKit:
            descriptors: List[str] (default None)
                List of RDKit descriptors to compute properties. When None, computes values
            for descriptors which are chosen based on options set in other arguments.
            use_fragment: bool, optional (default True)
                If True, the return value includes the fragment binary descriptors like 'fr_XXX'.
            ipc_avg: bool, optional (default True)
                If True, the IPC descriptor calculates with avg=True option.
                Please see this issue: https://github.com/rdkit/rdkit/issues/1527.
            is_normalized: bool, optional (default False)
                If True, the return value contains normalized features.
            use_bcut2d: bool, optional (default True)
                If True, the return value includes the descriptors like 'BCUT2D_XXX'.
            labels_only: bool, optional (default False)
                Returns only the presence or absence of a group.

        CircularFingerprint:
            radius: int, optional (default 2)
                Fingerprint radius.
            size: int, optional (default 2048)
                Length of generated bit vector.
            chiral: bool, optional (default False)
                Whether to consider chirality in fingerprint generation.
            bonds: bool, optional (default True)
                Whether to consider bond order in fingerprint generation.
            features: bool, optional (default False)
                Whether to use feature information instead of atom information; see
                RDKit docs for more info.
            sparse: bool, optional (default False)
                Whether to return a dict for each molecule containing the sparse
                fingerprint.
            smiles: bool, optional (default False)
                Whether to calculate SMILES strings for fragment IDs (only applicable
                when calculating sparse fingerprints).
            is_counts_based: bool, optional (default False)
                Whether to generates a counts-based fingerprint.

        Notes
        -----
        * If both `labels_only` and `is_normalized` are True, then `is_normalized` takes
            precendence and `labels_only` will not be applied.

        """
        try:
            from rdkit.Chem import Descriptors
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        self.use_fragment: bool = use_fragment
        self.use_bcut2d: bool = use_bcut2d
        self.is_normalized: bool = is_normalized
        self.ipc_avg: bool = ipc_avg
        self.labels_only = labels_only
        self.reqd_properties = {}
        self.normalized_desc: Dict[str, Callable] = {}

        all_descriptors = {name: func for name, func in Descriptors.descList}

        if not descriptors:
            # user has not specified a descriptor list
            for desc_name, function in all_descriptors.items():
                if self.use_fragment is False and desc_name.startswith("fr_"):
                    continue
                if self.use_bcut2d is False and desc_name.startswith("BCUT2D_"):
                    continue
                self.reqd_properties[desc_name] = function
        else:
            for desc_name in descriptors:
                if desc_name in all_descriptors:
                    self.reqd_properties[desc_name] = all_descriptors[desc_name]
                else:
                    logging.error("Unable to find specified property %s" % desc_name)

        # creates normalized functions dictionary if normalized features are required
        if is_normalized:
            self.normalized_desc = self._make_normalised_func_dict()
            desc_names = list(self.reqd_properties.keys())
            for desc_name in desc_names:
                if desc_name not in self.normalized_desc:
                    logger.warning(
                        "No normalization for %s. Feature removed!", desc_name
                    )
                    self.reqd_properties.pop(desc_name)

        self.reqd_properties = dict(sorted(self.reqd_properties.items()))

        self.radius = radius
        self.size = size
        self.chiral = chiral
        self.bonds = bonds
        self.features = features
        self.sparse = sparse
        self.smiles = smiles
        self.is_counts_based = is_counts_based
        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.size,
            includeChirality=self.chiral,
            useBondTypes=self.bonds,
            onlyNonzeroInvariants=self.features,
        )

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
        """
        Calculate RDKit descriptors.

        Parameters
        ----------
        datapoint: RDKitMol
            RDKit Mol object

        Returns
        -------
        np.ndarray
            1D array of RDKit descriptors for `mol`.
            The length is `len(self.descriptors)`.

        """
        features = []
        for desc_name, function in self.reqd_properties.items():
            if desc_name == "Ipc" and self.ipc_avg:
                feature = function(datapoint, avg=True)
            else:
                feature = function(datapoint)

            if self.is_normalized:
                # get cdf(feature) for that descriptor
                feature = self.normalized_desc[desc_name](feature)

            features.append(feature)

        np_features_rdkit = np.asarray(features)
        if self.labels_only:
            np.putmask(np_features_rdkit, np_features_rdkit != 0, 1)

        try:
            from rdkit import Chem, DataStructs
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")
        if "mol" in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )
        if self.sparse:
            info: Dict = {}
            circular_fp = self.mfpgen.GetFingerprint(
                datapoint,
            )
            circular_fp = circular_fp.GetNonzeroElements()  # convert to a dict

            # generate SMILES for fragments
            if self.smiles:
                fp_smiles = {}
                for fragment_id, count in circular_fp.items():
                    root, radius = info[fragment_id][0]
                    env = Chem.FindAtomEnvironmentOfRadiusN(datapoint, radius, root)
                    frag = Chem.PathToSubmol(datapoint, env)
                    smiles = Chem.MolToSmiles(frag)
                    fp_smiles[fragment_id] = {"smiles": smiles, "count": count}
                circular_fp = fp_smiles

        else:
            if self.is_counts_based:
                fp_sparse = self.mfpgen.GetCountFingerprint(
                    datapoint,
                )
                circular_fp = np.zeros(
                    (self.size,), dtype=float
                )  # initialise numpy array of zeros (shape: (required size,))
                DataStructs.ConvertToNumpyArray(fp_sparse, circular_fp)
            else:
                # fp = self.mfpgen.GetCountFingerprintAsNumPy(
                #     datapoint,
                # )
                fp_sparse = self.mfpgen.GetFingerprint(
                    datapoint,
                )
                circular_fp = np.zeros(
                    (self.size,), dtype=float
                )  # initialise numpy array of zeros (shape: (required size,))
                DataStructs.ConvertToNumpyArray(fp_sparse, circular_fp)

        combined_features = np.hstack((np_features_rdkit, circular_fp))

        return combined_features

    def _make_normalised_func_dict(self):
        """
        Helper function to create dictionary of RDkit descriptors and
        associated cumulative density functions (CDFs) to generate normalized features.

        -------------------------------------------------------------------------------
        -------------------------------------------------------------------------------
        Copyright (c) 2018-2021, Novartis Institutes for BioMedical Research Inc.
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are
        met:

        * Redistributions of source code must retain the above copyright
            notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above
            copyright notice, this list of conditions and the following
            disclaimer in the documentation and/or other materials provided
            with the distribution.
        * Neither the name of Novartis Institutes for BioMedical Research Inc.
            nor the names of its contributors may be used to endorse or promote
            products derived from this software without specific prior written
            permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
        "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
        LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
        A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
        OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
        LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
        DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
        THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        -------------------------------------------------------------------------------
        -------------------------------------------------------------------------------

        """
        normalized_desc = {}
        # get sequence of descriptor names and normalization parameters from DescriptorsNormalizationParameters class
        parameters = DNP.desc_norm_params.items()

        for desc_name, (distribution_name, params, minV, maxV, avg, std) in parameters:
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # get required distribution_ from `scipy.stats` module.
            cont_distribution = getattr(st, distribution_name)

            # cdf => cumulative density functions
            # make the cdf with the parameters.
            def norm_cdf(
                v: Union[int, float],
                distribution_: rv_continuous = cont_distribution,
                arg: Sequence[float] = arg,
                loc: float = loc,
                scale: float = scale,
                minV: float = minV,
                maxV: float = maxV,
            ) -> np.ndarray:
                v = distribution_.cdf(
                    np.clip(v, minV, maxV), loc=loc, scale=scale, *arg
                )
                return np.clip(v, 0.0, 1.0)

            normalized_desc[desc_name] = norm_cdf
        return normalized_desc
