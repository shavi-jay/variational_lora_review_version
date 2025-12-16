from typing import Union

from deepchem.models.optimizers import LearningRateSchedule
from deepchem.models.optimizers import Optimizer as DC_Optimizer

import torch_optimizer as optim

class Lamb(DC_Optimizer):
    """The Lamb optimization algorithm."""

    def __init__(
        self,
        learning_rate: Union[float, LearningRateSchedule] = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-08,
    ):
        """Construct an Lamb optimizer.

        Parameters
        ----------
        learning_rate: float or LearningRateSchedule
          the learning rate to use for optimization
        beta1: float
          a parameter of the Lamb algorithm
        beta2: float
          a parameter of the Lamb algorithm
        epsilon: float
          a parameter of the Lamb algorithm
        """
        super(Lamb, self).__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _create_pytorch_optimizer(self, params):
        if isinstance(self.learning_rate, LearningRateSchedule):
            lr = self.learning_rate.initial_rate
        else:
            lr = self.learning_rate
        return optim.Lamb(params, lr, (self.beta1, self.beta2), self.epsilon)
