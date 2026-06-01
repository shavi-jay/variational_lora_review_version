import math
from typing import TYPE_CHECKING

from src.variational_model import (
    VariationalMolformerSingle,
    VariationalMolbertSingle,
    VariationalMoleSingle,
)

if TYPE_CHECKING:
    from src.variational_inference.blob_layer import BLoBConfig


class _BLoBMixinMFVIBeta:
    """Cyclic exponential KL re-weighting (BLoB paper Appendix B.1).

    Mixed in before the VariationalModel base so that mfvi_beta is resolved
    from this class in MRO. All accessed attributes (_global_epoch,
    _global_step, mfvi_config, train_dataset_size, batch_size) are provided
    by VariationalModel at runtime.
    """

    if TYPE_CHECKING:
        _global_epoch: int
        _global_step: int
        mfvi_config: "BLoBConfig"
        train_dataset_size: int
        batch_size: int

    @property
    def M(self) -> int:
        """Number of mini-batches per epoch."""
        return math.floor(self.train_dataset_size / self.batch_size)

    @property
    def mfvi_beta(self) -> float:
        if self._global_epoch <= self.mfvi_config.max_likelihood_epochs:
            return 0.0
        if not self.mfvi_config.kl_reweighting:
            return 1.0 / self.M
        i = self._global_step % self.M
        if i == 0:
            i = self.M
        return (2 ** i) / (2 ** (self.M + 1) - 1)


class VariationalMolformerBLoBSingle(_BLoBMixinMFVIBeta, VariationalMolformerSingle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VariationalMolbertBLoBSingle(_BLoBMixinMFVIBeta, VariationalMolbertSingle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class VariationalMoleBLoBSingle(_BLoBMixinMFVIBeta, VariationalMoleSingle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
