"""
Adapted from DeepChem: deepchem.models.torch_models.hf_models.py
"""

import os

import logging
import time
from collections.abc import Sequence as SequenceCollection
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from deepchem.models.optimizers import LearningRateSchedule
from deepchem.models.torch_models import TorchModel
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.typing import LossFn, OneOrMany
from deepchem.data import Dataset
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.models.auto import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.configuration_roberta import RobertaConfig

logger = logging.getLogger(__name__)
logging.getLogger("deepchem").setLevel(logging.WARNING)

logger.setLevel(logging.WARNING)

if TYPE_CHECKING:
    import transformers
    from transformers.modeling_utils import PreTrainedModel

from src.training_utils import EarlyStopping, classification_error
from src.utils import short_timer, return_short_time


class PreTrainedChemBERTa(nn.Module):
    """
    Combining ChemBERTa pretrained with output layer
    """

    def __init__(
        self,
        pre_trained_model: "PreTrainedModel",
        config: RobertaConfig,
        device: torch.device,
        n_tasks: Optional[int] = None,
        problem_type: Optional[str] = None,
    ):
        super().__init__()
        self.pre_trained_model = pre_trained_model
        self.problem_type = config.problem_type
        self.num_labels = config.num_labels

        dim_pre_trained_last_hidden = list(self.pre_trained_model.parameters())[
            -1
        ].shape[-1]

        self.output_layer = nn.Linear(dim_pre_trained_last_hidden, self.num_labels)

        self.output_layer.to(device)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        hidden_outputs: SequenceClassifierOutput = self.pre_trained_model(
            input_ids,
            attention_mask,
            token_type_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        output_logits = self.output_layer(hidden_outputs.logits)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(output_logits.device)
            if self.problem_type is None:
                if self.num_labels == 1:
                    problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(output_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(output_logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    output_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(output_logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=output_logits,
        )


class HuggingFaceModel(TorchModel):
    r"""Wrapper class that wraps HuggingFace models as DeepChem models

    The class provides a wrapper for wrapping models from HuggingFace
    ecosystem in DeepChem and training it via DeepChem's api. The reason
    for this might be that you might want to do an apples-to-apples comparison
    between HuggingFace from the transformers library and DeepChem library.

    The `HuggingFaceModel` has a Has-A relationship by wrapping models from
    `transformers` library. Once a model is wrapped, DeepChem's API are used
    for training, prediction, evaluation and other downstream tasks.

    A `HuggingFaceModel` wrapper also has a `tokenizer` which tokenizes raw
    SMILES strings into tokens to be used by downstream models.  The SMILES
    strings are generally stored in the `X` attribute of deepchem.data.Dataset object'.
    This differs from the DeepChem standard workflow as tokenization is done
    on the fly here. The approach allows us to leverage `transformers` library's fast
    tokenization algorithms and other utilities like data collation, random masking of tokens
    for masked language model training etc.


    Parameters
    ----------
    model: transformers.modeling_utils.PreTrainedModel
        The HuggingFace model to wrap.
    task: str, (optional, default None)
        The task defines the type of learning task in the model. The supported tasks are
         - `mlm` - masked language modeling commonly used in pretraining
         - `mtr` - multitask regression - a task used for both pretraining base models and finetuning
         - `regression` - use it for regression tasks, like property prediction
         - `classification` - use it for classification tasks
        When the task is not specified or None, the wrapper returns raw output of the HuggingFaceModel.
        In cases where the HuggingFaceModel is a model without a task specific head, this output will be
        the last hidden states.
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer
        Tokenizer

    Example
    -------
    >>> import os
    >>> import tempfile
    >>> tempdir = tempfile.mkdtemp()

    >>> # preparing dataset
    >>> smiles = ['CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1', 'CC[NH+](CC)C1CCC([NH2+]C2CC2)(C(=O)[O-])C1', \
    ...     'COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC', 'OCCn1cc(CNc2cccc3c2CCCC3)nn1', \
    ...     'CCCCCCc1ccc(C#Cc2ccc(C#CC3=CC=C(CCC)CC3)c(C3CCCCC3)c2)c(F)c1', 'nO=C(NCc1ccc(F)cc1)N1CC=C(c2c[nH]c3ccccc23)CC1']
    >>> filepath = os.path.join(tempdir, 'smiles.txt')
    >>> f = open(filepath, 'w')
    >>> f.write('\n'.join(smiles))
    253
    >>> f.close()

    >>> # preparing tokenizer
    >>> from tokenizers import ByteLevelBPETokenizer
    >>> from transformers.models.roberta import RobertaTokenizerFast
    >>> tokenizer = ByteLevelBPETokenizer()
    >>> tokenizer.train(files=filepath, vocab_size=1_000, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    >>> tokenizer_path = os.path.join(tempdir, 'tokenizer')
    >>> os.makedirs(tokenizer_path)
    >>> result = tokenizer.save_model(tokenizer_path)
    >>> tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)

    >>> # preparing dataset
    >>> import pandas as pd
    >>> import deepchem as dc
    >>> smiles = ["CCN(CCSC)C(=O)N[C@@](C)(CC)C(F)(F)F","CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1"]
    >>> labels = [3.112,2.432]
    >>> df = pd.DataFrame(list(zip(smiles, labels)), columns=["smiles", "task1"])
    >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    ...     df.to_csv(tmpfile.name)
    ...     loader = dc.data.CSVLoader(["task1"], feature_field="smiles", featurizer=dc.feat.DummyFeaturizer())
    ...     dataset = loader.create_dataset(tmpfile.name)

    >>> # pretraining
    >>> from deepchem.models.torch_models.hf_models import HuggingFaceModel
    >>> from transformers.models.roberta import RobertaForMaskedLM, RobertaModel, RobertaConfig
    >>> config = RobertaConfig(vocab_size=tokenizer.vocab_size)
    >>> model = RobertaForMaskedLM(config)
    >>> hf_model = HuggingFaceModel(model=model, tokenizer=tokenizer, task='mlm', model_dir='model-dir')
    >>> training_loss = hf_model.fit(dataset, nb_epoch=1)

    >>> # finetuning a regression model
    >>> from transformers.models.roberta import RobertaForSequenceClassification
    >>> config = RobertaConfig(vocab_size=tokenizer.vocab_size, problem_type='regression', num_labels=1)
    >>> model = RobertaForSequenceClassification(config)
    >>> hf_model = HuggingFaceModel(model=model, tokenizer=tokenizer, task='regression', model_dir='model-dir')
    >>> hf_model.load_from_pretrained()
    >>> training_loss = hf_model.fit(dataset, nb_epoch=1)
    >>> prediction = hf_model.predict(dataset)  # prediction
    >>> eval_results = hf_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.mae_score))

    >>> # finetune a classification model
    >>> # making dataset suitable for classification
    >>> import numpy as np
    >>> y = np.random.choice([0, 1], size=dataset.y.shape)
    >>> dataset = dc.data.NumpyDataset(X=dataset.X, y=y, w=dataset.w, ids=dataset.ids)

    >>> from transformers import RobertaForSequenceClassification
    >>> config = RobertaConfig(vocab_size=tokenizer.vocab_size)
    >>> model = RobertaForSequenceClassification(config)
    >>> hf_model = HuggingFaceModel(model=model, task='classification', tokenizer=tokenizer)
    >>> training_loss = hf_model.fit(dataset, nb_epoch=1)
    >>> predictions = hf_model.predict(dataset)
    >>> eval_result = hf_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.f1_score))
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "transformers.tokenization_utils.PreTrainedTokenizer",
        task: Optional[str] = None,
        n_tasks: Optional[int] = None,
        problem_type: Optional[str] = None,
        is_truncation: bool = False,
        **kwargs,
    ):
        self.task = task
        self.n_tasks = n_tasks
        self.problem_type = (
            problem_type if problem_type is not None else self._get_problem_type()
        )
        self.tokenizer = tokenizer
        self.is_truncation = is_truncation
        if self.task == "mlm":
            self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
        else:
            self.data_collator = None  # type: ignore
        # Ignoring type. For TorchModel, loss is a required argument but HuggingFace computes
        # loss during the forward iteration, removing the need for a loss function.
        super(HuggingFaceModel, self).__init__(
            model=model, loss=None, **kwargs  # type: ignore
        )

    def _get_problem_type(self):
        if self.task == "mlm":
            problem_type = None
        elif self.task == "mtr":
            problem_type = "regression"
        elif self.task == "regression":
            problem_type = "regression"
        elif self.task == "classification":
            if self.n_tasks == 1:
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
        else:
            raise ValueError(f"Unknown problem type: {self.task}")
        return problem_type

    def load_from_pretrained(  # type: ignore
        self,
        model_dir: Optional[str] = None,
        from_hf_checkpoint: bool = False,
        **kwargs,
    ):
        """Load HuggingFace model from a pretrained checkpoint.

        The utility can be used for loading a model from a checkpoint.
        Given `model_dir`, it checks for existing checkpoint in the directory.
        If a checkpoint exists, the models state is loaded from the checkpoint.

        If the option `from_hf_checkpoint` is set as True, then it loads a pretrained
        model using HuggingFace models `from_pretrained` method. This option
        interprets model_dir as a model id of a pretrained model hosted inside a model repo
        on huggingface.co or path to directory containing model weights saved using `save_pretrained`
        method of a HuggingFace model.

        Parameter
        ----------
        model_dir: str
            Directory containing model checkpoint
        from_hf_checkpoint: bool, default False
            Loads a pretrained model from HuggingFace checkpoint.

        Example
        -------
        >>> from transformers import RobertaTokenizerFast
        >>> tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/PubChem10M_SMILES_BPE_60k")

        >>> from deepchem.models.torch_models.hf_models import HuggingFaceModel
        >>> from transformers.models.roberta import RobertaForMaskedLM, RobertaModel, RobertaConfig
        >>> config = RobertaConfig(vocab_size=tokenizer.vocab_size)
        >>> model = RobertaForMaskedLM(config)
        >>> pretrain_model = HuggingFaceModel(model=model, tokenizer=tokenizer, task='mlm', model_dir='model-dir')
        >>> pretrain_model.save_checkpoint()

        >>> from transformers import RobertaForSequenceClassification
        >>> config = RobertaConfig(vocab_size=tokenizer.vocab_size)
        >>> model = RobertaForSequenceClassification(config)
        >>> finetune_model = HuggingFaceModel(model=model, task='classification', tokenizer=tokenizer, model_dir='model-dir')

        >>> finetune_model.load_from_pretrained()
        """
        if model_dir is None:
            model_dir = self.model_dir

        if from_hf_checkpoint:
            # FIXME Transformers library has an api like AutoModel.from_pretrained. It allows to
            # initialise and create a model instance directly without requiring a class instance initialisation step.
            # To use `load_from_pretrained` in DeepChem, we need to follow a two step process
            # of initialising class instance and then loading weights via `load_from_pretrained`.
            if self.task == "mlm":
                self.model = AutoModelForMaskedLM.from_pretrained(
                    model_dir, num_labels=self.num_labels, **kwargs
                )
            elif self.task in ["mtr", "regression", "classification"]:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir, num_labels=self.num_labels, **kwargs
                )
            else:
                self.model = AutoModel.from_pretrained(
                    model_dir, num_labels=self.num_labels, **kwargs
                )

            self.model.config.update(
                {"problem_type": self.problem_type, "num_labels": self.n_tasks}
            )

        elif not from_hf_checkpoint:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            print(checkpoints)
            if len(checkpoints) == 0:
                raise ValueError("No checkpoint found")
            else:
                checkpoint = checkpoints[0]
                data = torch.load(checkpoint, map_location=self.device)
                self.model.load_state_dict(data["model_state_dict"], strict=False)
                if data.get("global_step") is not None:
                    self._pretrained_global_step = data.get("global_step")

        self.model.to(self.device)

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        smiles_batch, y, w = batch
        tokens = self.tokenizer(
            smiles_batch[0].tolist(),
            padding=True,
            return_tensors="pt",
            truncation=self.is_truncation,
        )

        if self.task == "mlm":
            inputs, labels = self.data_collator.torch_mask_tokens(tokens["input_ids"])
            inputs = {
                "input_ids": inputs.to(self.device),
                "labels": labels.to(self.device),
                "attention_mask": tokens["attention_mask"].to(self.device),
            }
            return inputs, None, w
        elif self.task in ["regression", "classification", "mtr"]:
            if y is not None:
                # y is None during predict
                y = torch.from_numpy(y[0])
                if self.task == "regression" or self.task == "mtr":
                    y = y.float().to(self.device)
                elif self.task == "classification":
                    y = y.long().to(self.device)
            for key, value in tokens.items():
                tokens[key] = value.to(self.device)

            inputs = {**tokens, "labels": y}
            return inputs, y, w

    def _ensure_built(self) -> None:
        """The first time this is called, create internal data structures."""
        if self._built:
            return
        self._built = True
        self._global_step = 0
        self._global_epoch = 0
        self._pytorch_optimizer = self.optimizer._create_pytorch_optimizer(
            self.model.parameters()
        )
        if isinstance(self.optimizer.learning_rate, LearningRateSchedule):
            self._lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                self._pytorch_optimizer
            )
        else:
            self._lr_schedule = None

    def rebuild(self) -> None:
        """Rebuild the model."""
        self._built = False
        self._ensure_built()

    def fit(
        self,
        dataset: Dataset,
        nb_epoch: int = 10,
        max_checkpoints_to_keep: int = 5,
        min_epochs_to_train: int = 0,
        checkpoint_interval: int = 1000,
        deterministic: bool = False,
        restore: bool = False,
        variables: Optional[List[torch.nn.Parameter]] = None,
        loss: Optional[LossFn] = None,
        callbacks: Union[Callable, List[Callable]] = [],
        all_losses: Optional[List[float]] = None,
        test_log_frequency: int = None,
        is_epoch_test_logging: bool = None,
        test_dataset: Dataset = None,
        max_test_batches: int | None = None,
        early_stopping: bool = True,
    ) -> float:
        """Train this model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            the Dataset to train on
        nb_epoch: int
            the number of epochs to train for
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        min_epochs_to_train: int
            the minimum number of epochs to train for, before early stopping can be applied
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
        deterministic: bool
            if True, the samples are processed in order.  If False, a different random
            order is used for each epoch.
        restore: bool
            if True, restore the model from the most recent checkpoint and continue training
            from there.  If False, retrain the model from scratch.
        variables: list of torch.nn.Parameter
            the variables to train.  If None (the default), all trainable variables in
            the model are used.
        loss: function
            a function of the form f(outputs, labels, weights) that computes the loss
            for each batch.  If None (the default), the model's standard loss function
            is used.
        callbacks: function or list of functions
            one or more functions of the form f(model, step) that will be invoked after
            every step.  This can be used to perform validation, logging, etc.
        all_losses: Optional[List[float]], optional (default None)
            If specified, all logged losses are appended into this list. Note that
            you can call `fit()` repeatedly with the same list and losses will
            continue to be appended.
        test_log_frequency: int (default None)
            the frequency at which test accuracy is logged. This is the number of global steps
            between each log. If None, then test loss is not logged during training.
        is_epoch_test_logging: bool (default False)
            specifies whether test loss should be logged at the end of the epoch. This can be
            set to True even if test_log_frequency is None
        test_dataset: Dataset (default None)
            the Dataset to test on
        max_test_batches: int (default None)
            maximum number of batches to evaluate on test dataset. If None then evaluate
            on full dataset.
        early_stopping: bool
            stop at the best checkpoint (amongst the list of checkpoints that are saved).
        Returns
        -------
        The average loss over the most recent checkpoint interval
        """
        self._train_size = len(dataset)

        return self.fit_generator(
            generator=self.epoch_generator(
                dataset, epochs=nb_epoch, deterministic=deterministic
            ),
            max_checkpoints_to_keep=max_checkpoints_to_keep,
            checkpoint_interval=checkpoint_interval,
            min_epochs_to_train=min_epochs_to_train,
            restore=restore,
            variables=variables,
            loss=loss,
            callbacks=callbacks,
            all_losses=all_losses,
            test_log_frequency=test_log_frequency,
            is_epoch_test_logging=is_epoch_test_logging,
            test_dataset=test_dataset,
            max_test_batches=max_test_batches,
            early_stopping=early_stopping,
        )

    def epoch_generator(
        self,
        dataset: Dataset,
        epochs: int = 1,
        mode: str = "fit",
        deterministic: bool = True,
        pad_batches: bool = True,
    ) -> Iterable[Tuple[List, List, List]]:
        """Create a generator that iterates batches for a dataset.

        Subclasses may override this method to customize how model inputs are
        generated from the data.

        Parameters
        ----------
        dataset: Dataset
        the data to iterate
        epochs: int
        the number of times to iterate over the full dataset
        mode: str
        allowed values are 'fit' (called during training), 'predict' (called
        during prediction), and 'uncertainty' (called during uncertainty
        prediction)
        deterministic: bool
        whether to iterate over the dataset in order, or randomly shuffle the
        data for each epoch
        pad_batches: bool
        whether to pad each batch up to this model's preferred batch size

        Returns
        -------
        a generator that iterates batches, each represented as a tuple of lists:
        ([inputs], [outputs], [weights])
        """
        for epoch in range(epochs):
            for X_b, y_b, w_b, ids_b in dataset.iterbatches(
                batch_size=self.batch_size,
                deterministic=deterministic,
                pad_batches=pad_batches,
            ):
                yield ([X_b], [y_b], [w_b])
            self._global_epoch += 1

    def fit_generator(
        self,
        generator: Iterable[Tuple[Any, Any, Any]],
        max_checkpoints_to_keep: int = 5,
        min_epochs_to_train: int = 0,
        checkpoint_interval: int = 1000,
        restore: bool = False,
        variables: Optional[
            Union[List[torch.nn.Parameter], torch.nn.ParameterList]
        ] = None,
        loss: Optional[LossFn] = None,
        callbacks: Union[Callable, List[Callable]] = [],
        all_losses: Optional[List[float]] = None,
        test_log_frequency: int = None,
        is_epoch_test_logging: bool = None,
        test_dataset: Dataset = None,
        max_test_batches: int = None,
        early_stopping: bool = True,
    ) -> float:
        """Train this model on data from a generator.

        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        min_epochs_to_train: int
            the minimum number of epochs to train for, before early stopping can be applied
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
        restore: bool
            if True, restore the model from the most recent checkpoint and continue training
            from there.  If False, retrain the model from scratch.
        variables: list of torch.nn.Parameter
            the variables to train.  If None (the default), all trainable variables in
            the model are used.
        loss: function
            a function of the form f(outputs, labels, weights) that computes the loss
            for each batch.  If None (the default), the model's standard loss function
            is used.
        callbacks: function or list of functions
            one or more functions of the form f(model, step) that will be invoked after
            every step.  This can be used to perform validation, logging, etc.
        all_losses: Optional[List[float]], optional (default None)
            If specified, all logged losses are appended into this list. Note that
            you can call `fit()` repeatedly with the same list and losses will
            continue to be appended.
        test_log_frequency: int (default None)
            the frequency at which test accuracy is logged. This is the number of global steps
            between each log. If None, then test loss is not logged during training.
        is_epoch_test_logging: bool (default False)
            specifies whether test loss should be logged at the end of the epoch. This can be
            set to True even if test_log_frequency is None
        test_dataset: Dataset (default None)
            the Dataset to test on
        max_test_batches: int (default None)
            maximum number of batches to evaluate on test dataset. If None then evaluate
            on full dataset.
        early_stopping: bool
            stop at the best checkpoint (amongst the list of checkpoints that are saved).
        Returns
        -------
        The average loss over the most recent checkpoint interval

        Note
        ----
        A HuggingFace model can return embeddings (last hidden state), attentions.
        Support must be added to return the embeddings to the user, so that it can
        be used for other downstream applications.
        """
        if not isinstance(callbacks, SequenceCollection):
            callbacks = [callbacks]
        self._ensure_built()
        self.model.train()
        self.num_labels = self.model.num_labels
        self.total_iteration_time = 0
        avg_loss = 0.0
        last_avg_loss = 0.0
        averaged_batches = 0
        self._create_optimizer_and_lr_scheduler(variables)
        time1 = time.time()

        if self.problem_type == "single_label_classification":
            avg_error = 0.0

        self._check_test_dataset_specified(
            test_dataset, is_epoch_test_logging, test_log_frequency, checkpoint_interval
        )

        current_epoch = self._global_epoch - 1

        if not hasattr(self, "early_stopper"):
            self.early_stopper = EarlyStopping(
                max_checkpoints_to_keep, early_stopping, min_epochs_to_train
            )
        elif self.early_stopper is None:
            self.early_stopper = EarlyStopping(
                max_checkpoints_to_keep, early_stopping, min_epochs_to_train
            )

        if self.early_stopper.stopped:
            return
        
        # Main training loop.

        for batch in generator:
            if restore:
                self.restore()
                restore = False
            inputs: OneOrMany[torch.Tensor]

            with return_short_time() as timer:
                inputs, labels, weights = self._prepare_batch(batch)

                self._pytorch_optimizer.zero_grad()
                outputs = self.model(**inputs)

                if self._loss_outputs is not None:
                    outputs = [outputs[i] for i in self._loss_outputs]
                batch_loss = self.loss_function(outputs.get("loss"))
                batch_loss.backward()
                self._pytorch_optimizer.step()
            self.total_iteration_time += timer.time

            if self._lr_schedule is not None:
                self._lr_schedule.step()
            self._global_step += 1
            current_step = self._global_step

            avg_loss += batch_loss

            if self.problem_type == "single_label_classification" and (
                self.wandb_logger is not None
            ):
                batch_error = classification_error(
                    outputs.get("logits"), inputs.get("labels"), self.num_labels
                )
                avg_error += batch_error

            # Report progress and write checkpoints.
            averaged_batches += 1
            should_log = current_step % self.log_frequency == 0
            if should_log:
                avg_loss = float(avg_loss) / averaged_batches
                logger.info(
                    "Ending global_step %d: Average loss %g" % (current_step, avg_loss)
                )
                if all_losses is not None:
                    all_losses.append(avg_loss)
                # Capture the last avg_loss in case of return since we're resetting to 0 now
                last_avg_loss = avg_loss
                avg_loss = 0.0
                averaged_batches = 0

            if (
                checkpoint_interval > 0
                and current_step % checkpoint_interval == checkpoint_interval - 1
            ):
                test_avg_loss = self._evaluate_and_log_on_test(
                    test_dataset, is_checkpoint_evaluation=True
                )
                if self.early_stopper.should_early_stop(
                    test_avg_loss, self._global_epoch
                ):
                    break
                else:
                    self.save_checkpoint(max_checkpoints_to_keep)
            for c in callbacks:
                c(self, current_step)
            if self.tensorboard and should_log:
                self._log_scalar_to_tensorboard("loss", batch_loss, current_step)
            if (self.wandb_logger is not None) and should_log:
                all_data = dict({"train/loss": batch_loss})
                if self.problem_type == "single_label_classification":
                    all_data.update({"train/classification error": batch_error})
                self.wandb_logger.log_data(all_data, step=current_step)

            if test_log_frequency is None:
                should_log_test = False
            else:
                should_log_test = current_step % test_log_frequency == 0
            if should_log_test:
                self._evaluate_and_log_on_test(test_dataset, max_test_batches)

            new_epoch = current_epoch != self._global_epoch
            if new_epoch:
                if is_epoch_test_logging:
                    self._evaluate_and_log_on_test(
                        test_dataset, max_test_batches, is_epoch_evaluation=True
                    )
                if self.wandb_logger is not None:
                    all_data = dict(
                        {"train/epoch loss": batch_loss, "epoch": self._global_epoch}
                    )
                    self.wandb_logger.log_data(all_data, step=current_step)
                current_epoch = self._global_epoch

        # Report final results.
        if averaged_batches > 0:
            avg_loss = float(avg_loss) / averaged_batches
            logger.info(
                "Ending global_step %d: Average loss %g" % (current_step, avg_loss)
            )
            if all_losses is not None:
                all_losses.append(avg_loss)
            last_avg_loss = avg_loss

        if checkpoint_interval > 0:
            if self.early_stopper.stopped:
                self.restore(checkpoint_number=self.early_stopper.optimal_checkpoint)
                print(
                    f"Early stopping at {self.get_global_step()} steps; {self._global_epoch} epochs"
                )
            else:
                final_loss = self._evaluate_and_log_on_test(
                    test_dataset, is_checkpoint_evaluation=True
                )
                if self.early_stopper.should_early_stop(final_loss, self._global_epoch):
                    self.restore(
                        checkpoint_number=self.early_stopper.optimal_checkpoint
                    )
                else:
                    self.save_checkpoint(max_checkpoints_to_keep)
                    self.restore(
                        checkpoint_number=self.early_stopper.optimal_checkpoint
                    )
            self.save_checkpoint(max_checkpoints_to_keep + 1)

        time2 = time.time()
        logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
        return last_avg_loss

    @staticmethod
    def _check_test_dataset_specified(
        test_dataset: Dataset,
        is_epoch_test_logging: bool,
        test_log_frequency: int,
        checkpoint_interval: int,
    ):
        if test_dataset is None:
            if is_epoch_test_logging == True:
                raise ValueError(
                    "Must specify test_dataset if is_epoch_test_logging is True"
                )
            if test_log_frequency is not None:
                raise ValueError(
                    "Must specify test_dataset if test_log_frequency specified"
                )
            if checkpoint_interval > 0:
                raise ValueError(
                    "Must specify test_dataset if checkpoint_interval specified"
                )

    def _create_optimizer_and_lr_scheduler(
        self,
        variables: Optional[
            Union[List[torch.nn.Parameter], torch.nn.ParameterList]
        ] = None,
    ):
        if variables is not None:
            var_key = tuple(variables)
            if var_key in self._optimizer_for_vars:
                self._pytorch_optimizer, self._lr_schedule = self._optimizer_for_vars[
                    var_key
                ]
            else:
                self._pytorch_optimizer = self.optimizer._create_pytorch_optimizer(
                    variables
                )
                if isinstance(self.optimizer.learning_rate, LearningRateSchedule):
                    self._lr_schedule = (
                        self.optimizer.learning_rate._create_pytorch_schedule(
                            self._pytorch_optimizer
                        )
                    )
                else:
                    self._lr_schedule = None
                self._optimizer_for_vars[var_key] = (
                    self._pytorch_optimizer,
                    self._lr_schedule,
                )
        return

    def loss_function(self, batch_loss: torch.Tensor, **kwargs):
        return batch_loss

    def _test_loss_function(
        self,
        batch_loss: torch.Tensor,
        **kwargs,
    ):
        return self.loss_function(batch_loss=batch_loss, **kwargs)

    def _evaluate_on_test(
        self,
        generator: Iterable[Tuple[List, List, List]],
        max_test_batches: int | None = None,
        is_epoch_evaluation: bool = False,
        is_checkpoint_evaluation: bool = False,
    ):
        self.model.eval()
        sum_avg_loss = 0.0
        sum_avg_err = 0.0
        averaged_batches = 0

        with torch.no_grad():
            for batch in generator:
                inputs: OneOrMany[torch.Tensor]
                inputs, _, _ = self._prepare_batch(batch)
                output_values = self.model(**inputs)
                batch_loss = self._test_loss_function(
                    output_values.get("loss"),
                    is_epoch_evaluation=is_epoch_evaluation,
                    is_checkpoint_evaluation=is_checkpoint_evaluation,
                )

                sum_avg_loss += batch_loss
                averaged_batches += 1

                if self.problem_type == "single_label_classification":
                    batch_error = classification_error(
                        output_values.get("logits"),
                        inputs.get("labels"),
                        self.num_labels,
                    )
                    sum_avg_err += batch_error

                if max_test_batches is not None:
                    if averaged_batches >= max_test_batches:
                        break

            avg_loss = float(sum_avg_loss) / averaged_batches
            if self.problem_type == "single_label_classification":
                avg_err = float(sum_avg_err) / averaged_batches
            else:
                avg_err = None
            return avg_loss, avg_err

    def _evaluate_and_log_on_test(
        self,
        test_dataset: Dataset,
        max_test_batches: int | None = None,
        is_epoch_evaluation: bool = False,
        is_checkpoint_evaluation: bool = False,
        log_output: bool = True,
    ):
        self.model.eval()

        generator = self.default_generator(
            dataset=test_dataset,
            epochs=1,
            mode="predict",
            deterministic=False,
            pad_batches=False,
        )

        with torch.no_grad():
            avg_loss, avg_err = self._evaluate_on_test(
                generator=generator,
                max_test_batches=max_test_batches,
                is_epoch_evaluation=is_epoch_evaluation,
                is_checkpoint_evaluation=is_checkpoint_evaluation,
            )

            if (self.wandb_logger is not None) and (log_output == True):
                if is_epoch_evaluation:
                    evalutation_type_str = "epoch "
                elif is_checkpoint_evaluation:
                    evalutation_type_str = "checkpoint "
                else:
                    evalutation_type_str = ""
                all_data = dict({f"test/{evalutation_type_str}loss": avg_loss})
                if self.problem_type == "single_label_classification":
                    all_data.update(
                        {f"test/{evalutation_type_str}classification error": avg_err}
                    )
                self.wandb_logger.log_data(all_data, step=self.get_global_step())

        return avg_loss

    def _predict(
        self,
        generator: Iterable[Tuple[Any, Any, Any]],
        transformers: List[Transformer],
        uncertainty: bool,
        other_output_types: Optional[OneOrMany[str]],
    ):
        """Predicts output for data provided by generator.

        This is the private implementation of prediction. Do not
        call it directly. Instead call one of the public prediction methods.

        Parameters
        ----------
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        transformers: list of dc.trans.Transformers
            Transformers that the input data has been transformed by.  The output
            is passed through these transformers to undo the transformations.
        uncertainty: bool
            specifies whether this is being called as part of estimating uncertainty.
            If True, it sets the training flag so that dropout will be enabled, and
            returns the values of the uncertainty outputs.
        other_output_types: list, optional
            Provides a list of other output_types (strings) to predict from model.

        Returns
        -------
            a NumPy array of the model produces a single output, or a list of arrays
            if it produces multiple outputs

        Note
        ----
        A HuggingFace model does not output uncertainity. The argument is here
        since it is also present in TorchModel. Similarly, other variables like
        other_output_types are also not used. Instead, a HuggingFace model outputs
        loss, logits, hidden state and attentions.
        """
        results: Optional[List[List[np.ndarray]]] = None
        variances: Optional[List[List[np.ndarray]]] = None
        if uncertainty and (other_output_types is not None):
            raise ValueError(
                "This model cannot compute uncertainties and other output types simultaneously. Please invoke one at a time."
            )
        if uncertainty:
            if self._variance_outputs is None or len(self._variance_outputs) == 0:
                raise ValueError("This model cannot compute uncertainties")
            if len(self._variance_outputs) != len(self._prediction_outputs):
                raise ValueError(
                    "The number of variances must exactly match the number of outputs"
                )
        if other_output_types:
            if self._other_outputs is None or len(self._other_outputs) == 0:
                raise ValueError(
                    "This model cannot compute other outputs since no other output_types were specified."
                )
        self._ensure_built()
        self.model.eval()
        for batch in generator:
            inputs, labels, weights = batch
            inputs, _, _ = self._prepare_batch((inputs, None, None))

            # Invoke the model.
            output_values = self._predict_output_values(inputs)

            if isinstance(output_values, torch.Tensor):
                output_values = [output_values]
            output_values = [t.detach().cpu().numpy() for t in output_values]
            # Apply tranformers and record results.
            if uncertainty:
                var = [output_values[i] for i in self._variance_outputs]
                if variances is None:
                    variances = [var]
                else:
                    for i, t in enumerate(var):
                        variances[i].append(t)
            access_values = []
            if other_output_types:
                access_values += self._other_outputs
            elif self._prediction_outputs is not None:
                access_values += self._prediction_outputs

            if len(access_values) > 0:
                output_values = [output_values[i] for i in access_values]

            if len(transformers) > 0:
                if len(output_values) > 1:
                    raise ValueError(
                        "predict() does not support Transformers for models with multiple outputs."
                    )
                elif len(output_values) == 1:
                    output_values = [undo_transforms(output_values[0], transformers)]
            if results is None:
                results = [[] for i in range(len(output_values))]
            for i, t in enumerate(output_values):
                results[i].append(t)

        # Concatenate arrays to create the final results.
        final_results = []
        final_variances = []
        if results is not None:
            for r in results:
                final_results.append(np.concatenate(r, axis=0))

        if uncertainty and variances is not None:
            for v in variances:
                final_variances.append(np.concatenate(v, axis=0))
            return zip(final_results, final_variances)

        if len(final_results) == 1:
            return final_results[0]
        else:
            return np.array(final_results)

    def _predict_output_values(self, inputs, **kwargs):
        output_values = self.model(**inputs)
        output_values = output_values.get("logits")
        return output_values

    def restore(
        self,
        checkpoint: Optional[str] = None,
        checkpoint_number: int = 1,
        model_dir: Optional[str] = None,
        strict: bool = True,
        load_optimizer: bool = True,
    ) -> None:
        """Reload the values of all variables from a checkpoint file.

        Parameters
        ----------
        checkpoint: str
            the path to the checkpoint file to load.  If this is None, the checkpoint
            number given will be chosen automatically.  Call get_checkpoints() to get a
            list of all available checkpoints.
        checkpoint_number: int
            loads the numbered checkpoint. The default value is 1, which is the most
            recent checkpoint.
        model_dir: str, default None
            Directory to restore checkpoint from. If None, use self.model_dir.  If
            checkpoint is not None, this is ignored.
        """
        self._ensure_built()
        if checkpoint is None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
        if len(checkpoints) == 0:
            raise ValueError("No checkpoint found")
        if checkpoint_number > len(checkpoints):
            raise ValueError("Checkpoint does not exist")
        if checkpoint_number < 1:
            raise ValueError("Checkpoint number must be positive")
        checkpoint = checkpoints[checkpoint_number - 1]
        data = torch.load(checkpoint)
        self.model.load_state_dict(data["model_state_dict"], strict=strict)
        if load_optimizer:
            self._pytorch_optimizer.load_state_dict(data["optimizer_state_dict"])
        self._global_step = data["global_step"]
        self._global_epoch = data.get("global_epoch", 0)

    def save_checkpoint(
        self, max_checkpoints_to_keep: int = 5, model_dir: Optional[str] = None
    ) -> None:
        """Save a checkpoint to disk.

        Usually you do not need to call this method, since fit() saves checkpoints
        automatically.  If you have disabled automatic checkpointing during fitting,
        this can be called to manually write checkpoints.

        Parameters
        ----------
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        model_dir: str, default None
            Model directory to save checkpoint to. If None, revert to self.model_dir
        """
        self._ensure_built()
        if model_dir is None:
            model_dir = self.model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the checkpoint to a file.

        data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self._pytorch_optimizer.state_dict(),
            "global_step": self._global_step,
            "global_epoch": self._global_epoch,
        }
        temp_file = os.path.join(model_dir, "temp_checkpoint.pt")
        torch.save(data, temp_file)

        # Rename and delete older files.

        paths = [
            os.path.join(model_dir, "checkpoint%d.pt" % (i + 1))
            for i in range(max_checkpoints_to_keep)
        ]
        if os.path.exists(paths[-1]):
            os.remove(paths[-1])
        for i in reversed(range(max_checkpoints_to_keep - 1)):
            if os.path.exists(paths[i]):
                os.rename(paths[i], paths[i + 1])
        os.rename(temp_file, paths[0])
