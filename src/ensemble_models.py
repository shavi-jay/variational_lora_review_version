import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal, List, Callable, Iterable, Any
from collections.abc import Sequence as SequenceCollection
import logging
import time

import torch
import torch.nn as nn

from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaForSequenceClassification,
)


from deepchem.data import Dataset
from deepchem.utils.typing import LossFn, OneOrMany
from deepchem.models.optimizers import LearningRateSchedule

from src.deepchem_hf_models import HuggingFaceModel

from src.likelihood_model import LikelihoodSequenceClassifierOutput

from src.uncertainty_quantification_regression import (
    UncertaintyRegressionPredictionOutput,
    UncertaintyRegressionModel,
)

from src.model_molformer import (
    MolformerConfig,
    MolformerForSequenceClassification,
    MolformerForSequenceClassificationLikelihoodLoss,
)
from src.model_molbert import (
    MolbertConfig,
    MolbertForSequenceClassification,
    MolbertForSequenceClassificationLikelihoodLoss,
)
from src.model_mole import (
    MolEExtraConfig,
    MolEForSequenceClassification,
    MolEForSequenceClassificationLikelihoodLoss,
)

from src.training_utils import EarlyStopping, create_tokenizer

import wandb

logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger(__name__)

logger.setLevel(logging.WARNING)


@dataclass
class BasicEnsembleModelConfig:
    ensemble_model_type: str
    ensemble_member_config: RobertaConfig | MolformerConfig | MolbertConfig
    ensemble_size: int
    num_labels: int
    sequence_classifier_type: Literal["likelihood", "default"] = "default"


class BasicEnsembleModel(UncertaintyRegressionModel):
    """Ensemble of HF models for regression tasks."""

    def __init__(
        self,
        config: BasicEnsembleModelConfig,
    ):
        super().__init__()
        self.config = config
        self.ensemble_model_type = config.ensemble_model_type
        self.config.ensemble_member_config.num_labels = self.config.num_labels
        self.member_names = [f"member_{i}" for i in range(self.config.ensemble_size)]
        self.member_config = self.config.ensemble_member_config

        self._check_ensemble_model_type_with_config()

        self._create_pretrained_model()

        self.is_pretrained_model_set = False
        self.is_model_initialized = False

    def _check_ensemble_model_type_with_config(self):
        if isinstance(self.config.ensemble_member_config, RobertaConfig):
            config_type = "chemberta"
        elif isinstance(self.config.ensemble_member_config, MolformerConfig):
            config_type = "molformer"
        elif isinstance(self.config.ensemble_member_config, MolbertConfig):
            config_type = "molbert"
        elif isinstance(self.config.ensemble_member_config, MolEExtraConfig):
            config_type = "mole"
        else:
            raise ValueError(
                f"Unrecognised config type: {type(self.config.ensemble_member_config)}"
            )
        assert (
            self.ensemble_model_type == config_type
        ), "Ensemble model type and config do not match"

    def _create_pretrained_model(self):
        if self.ensemble_model_type == "chemberta":
            pretrained_models: nn.ModuleDict = nn.ModuleDict(
                {
                    member_name: RobertaForSequenceClassification(self.member_config)
                    for member_name in self.member_names
                }
            )
        elif self.ensemble_model_type == "molformer" and isinstance(
            self.member_config, MolformerConfig
        ):
            if self.config.sequence_classifier_type == "likelihood":
                model_class = MolformerForSequenceClassificationLikelihoodLoss
            elif self.config.sequence_classifier_type == "default":
                model_class = MolformerForSequenceClassification
            else:
                raise ValueError(
                    f"Unrecognized sequence_classification_type: {self.config.sequence_classifier_type}"
                )
            pretrained_models: nn.ModuleDict = nn.ModuleDict(
                {
                    member_name: model_class(self.member_config)
                    for member_name in self.member_names
                }
            )
        elif self.ensemble_model_type == "molbert" and isinstance(
            self.member_config, MolbertConfig
        ):
            if self.config.sequence_classifier_type == "likelihood":
                model_class = MolbertForSequenceClassificationLikelihoodLoss
            elif self.config.sequence_classifier_type == "default":
                model_class = MolbertForSequenceClassification
            else:
                raise ValueError(
                    f"Unrecognized sequence_classifier_type: {self.config.sequence_classifier_type}"
                )
            pretrained_models: nn.ModuleDict = nn.ModuleDict(
                {
                    member_name: model_class(self.member_config)
                    for member_name in self.member_names
                }
            )
        elif self.ensemble_model_type == "mole" and isinstance(
            self.member_config, MolEExtraConfig
        ):
            if self.config.sequence_classifier_type == "likelihood":
                model_class = MolEForSequenceClassificationLikelihoodLoss
            elif self.config.sequence_classifier_type == "default":
                model_class = MolEForSequenceClassification
            else:
                raise ValueError(
                    f"Unrecognized sequence_classifier_type: {self.config.sequence_classifier_type}"
                )
            pretrained_models: nn.ModuleDict = nn.ModuleDict(
                {
                    member_name: model_class(self.member_config)
                    for member_name in self.member_names
                }
            )
        else:
            raise ValueError(
                f"Unrecognized ensemble_model_type or member_config: {self.ensemble_model_type}, {type(self.member_config)}"
            )

        self.pretrained_models = pretrained_models

    def from_default_no_finetune(self, pretrained_model_path: str):
        if not isinstance(self.pretrained_models, nn.ModuleDict):
            raise ValueError(
                "Pretrained model must be a nn.ModuleDict for ensemble models"
            )
        if self.ensemble_model_type == "chemberta":
            for member_name in self.member_names:
                self.pretrained_models[member_name] = (
                    RobertaForSequenceClassification.from_pretrained(
                        pretrained_model_path, num_labels=self.member_config.num_labels
                    )
                )
        if self.ensemble_model_type == "molformer":
            if self.config.sequence_classifier_type == "likelihood":
                # print("LIKELIHOOD LOSS")
                model_class = MolformerForSequenceClassificationLikelihoodLoss
            elif self.config.sequence_classifier_type == "default":
                # print("DEFAULT LOSS")
                model_class = MolformerForSequenceClassification
            else:
                raise ValueError(
                    f"Unrecognized sequence_classification_type: {self.config.sequence_classifier_type}"
                )
            for member_name in self.member_names:
                if not isinstance(self.member_config, MolformerConfig):
                    raise ValueError(
                        "Member config must be MolformerConfig for Molformer ensemble"
                    )
                single_model = model_class(self.member_config)
                single_model.load_from_pretrained(pretrained_model_path)
                self.pretrained_models[member_name] = single_model
        if self.ensemble_model_type == "molbert":
            if self.config.sequence_classifier_type == "likelihood":
                model_class = MolbertForSequenceClassificationLikelihoodLoss
            elif self.config.sequence_classifier_type == "default":
                model_class = MolbertForSequenceClassification
            else:
                raise ValueError(
                    f"Unrecognized sequence_classifier_type: {self.config.sequence_classifier_type}"
                )
            for member_name in self.member_names:
                if not isinstance(self.member_config, MolbertConfig):
                    raise ValueError(
                        "Member config must be MolbertConfig for Molbert ensemble"
                    )
                single_model = model_class(self.member_config)
                single_model.load_from_pretrained(pretrained_model_path)
                self.pretrained_models[member_name] = single_model
        if self.ensemble_model_type == "mole":
            if self.config.sequence_classifier_type == "likelihood":
                model_class = MolEForSequenceClassificationLikelihoodLoss
            elif self.config.sequence_classifier_type == "default":
                model_class = MolEForSequenceClassification
            else:
                raise ValueError(
                    f"Unrecognized sequence_classification_type: {self.config.sequence_classifier_type}"
                )
            for member_name in self.member_names:
                if not isinstance(self.member_config, MolEExtraConfig):
                    raise ValueError(
                        "Member config must be MolEExtraConfig for MolE ensemble"
                    )
                single_model = model_class(self.member_config)
                single_model.load_from_pretrained(pretrained_model_path)
                self.pretrained_models[member_name] = single_model
        self.is_pretrained_model_set = True

    def set_active_member(self, active_member_name: str):
        for member_name, model in self.pretrained_models.items():
            if member_name == active_member_name:
                for name, param in model.named_parameters():
                    param.requires_grad = name in self.trainable_params[member_name]
            else:
                self.pretrained_models[member_name].requires_grad_(False)

        self.active_member = active_member_name

    def set_default_member(self):
        self.set_active_member(self.member_names[0])

    def set_trainable_parameters(self):
        if not self.check_model_initialized():
            raise RuntimeError(
                "Model not initialized. Call from_default_no_finetune first."
            )
        self.trainable_params = {}
        for member_name, model in self.pretrained_models.items():
            trainable = [
                name for name, param in model.named_parameters() if param.requires_grad
            ]
            self.trainable_params[member_name] = trainable

    def forward(
        self, *args, **kwargs
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        active_member = self.active_member

        return self.pretrained_models[active_member](*args, **kwargs)

    def check_model_initialized(self):
        if self.is_model_initialized == True:
            return True
        if self.is_pretrained_model_set == False:
            raise RuntimeError("Pretrianed model not set")
        if self.is_model_initialized == False:
            logger.info("Model Initialised!")
            self.is_model_initialized = True
        return True

    @torch.no_grad
    def predict_mean_and_std(
        self,
        labels: Optional[torch.LongTensor] = None,
        compute_std: bool = False,
        **input_kwargs,
    ) -> UncertaintyRegressionPredictionOutput:
        with torch.no_grad():
            self.eval()
            if self.config.sequence_classifier_type == "default":
                model_prediction = []
                for member_name in self.member_names:
                    self.set_active_member(member_name)
                    output = self.forward(
                        **input_kwargs,
                    )
                    if not isinstance(output, SequenceClassifierOutput):
                        raise ValueError(
                            f"Expected output type SequenceClassifierOutput, but got {type(output)}"
                        )
                    model_prediction.append(output.logits)

                model_prediction = torch.hstack(model_prediction)

                ensemble_mean = torch.mean(model_prediction, dim=1)

                ensemble_std = None

                if compute_std:
                    ensemble_mean_difference_squared = torch.pow(
                        model_prediction.T - ensemble_mean, 2
                    )

                    ensemble_sample_variance = torch.sum(
                        ensemble_mean_difference_squared, dim=0
                    )

                    ensemble_sample_variance = torch.div(
                        ensemble_sample_variance, self.config.ensemble_size - 1
                    )

                    ensemble_std = torch.sqrt(ensemble_sample_variance)

                return UncertaintyRegressionPredictionOutput(
                    mean=ensemble_mean, std=ensemble_std
                )
            elif self.config.sequence_classifier_type == "likelihood":
                model_prediction_mean = []
                model_prediction_std = []

                for member_name in self.member_names:
                    self.set_active_member(member_name)
                    model_output = self.forward(
                        **input_kwargs,
                    )
                    if not isinstance(model_output, LikelihoodSequenceClassifierOutput):
                        raise ValueError(
                            f"Expected output type LikelihoodSequenceClassifierOutput, but got {type(model_output)}"
                        )
                    model_prediction_mean.append(model_output.logits.unsqueeze(1))
                    if model_output.std_logits is None:
                        raise ValueError(
                            "std_logits is None. Ensure that the model is configured to output std_logits."
                        )
                    model_prediction_std.append(model_output.std_logits.unsqueeze(1))

                model_prediction_mean = torch.hstack(model_prediction_mean)
                model_prediction_std = torch.hstack(model_prediction_std)

                ensemble_mean = torch.mean(model_prediction_mean, dim=1)
                mean_predicted_var = torch.mean(model_prediction_std.pow(2), dim=1)

                ensemble_std = None

                if compute_std:
                    ensemble_mean_difference_squared = torch.pow(
                        model_prediction_mean.T - ensemble_mean, 2
                    )

                    ensemble_mean_sample_variance = torch.sum(
                        ensemble_mean_difference_squared, dim=0
                    )

                    ensemble_mean_sample_variance = torch.div(
                        ensemble_mean_sample_variance,
                        self.config.ensemble_size - 1,
                    )

                    ensemble_std = torch.sqrt(
                        ensemble_mean_sample_variance + mean_predicted_var
                    )

                return UncertaintyRegressionPredictionOutput(
                    mean=ensemble_mean, std=ensemble_std
                )
            else:
                raise ValueError(
                    f"Unrecognized sequence_classifier_type: {self.config.sequence_classifier_type}"
                )


class DeepChemBasicEnsembleModel(HuggingFaceModel):
    """Asyc training of model - similar to LoRA ensemble"""

    def __init__(
        self,
        model_config: BasicEnsembleModelConfig,
        task: Optional[str] = None,
        is_truncation: bool = False,
        **kwargs,
    ):
        self.task = task
        self.model_config = model_config
        self.is_truncation = is_truncation
        self.data_collator = None
        super(DeepChemBasicEnsembleModel, self).__init__(
            model=BasicEnsembleModel(model_config),
            task=task,
            tokenizer=create_tokenizer(model_config.ensemble_model_type),
            **kwargs,
        )
        self.ensemble_size = model_config.ensemble_size
        self.problem_type = self._get_problem_type()

    def load_from_pretrained(
        self,
        model_dir: Optional[str] = None,
        from_no_finetune: bool = False,
        from_local_checkpoint: bool = False,
        **kwargs,
    ):
        """Load HuggingFace model from a pretrained checkpoint.

        If the option `from_hf_checkpoint` is set as True, then it loads a pretrained
        model using HuggingFace models `from_pretrained` method. This option
        interprets model_dir as a model id of a pretrained model hosted inside a model repo
        on huggingface.co or path to directory containing model weights saved using `save_pretrained`
        method of a HuggingFace model.

        Parameter
        ----------
        model_dir: str
            Directory containing model checkpoint or name of huggingface checkpoint
        from_no_finetune: bool, default False
            Loads a pretrained model from a model that has not been finetune.
            If ChemBERTa this is from a HuggingFace checkpoint.
            If Molformer this is from local directory.
        from_local_checkpoint: bool, default True
            Loads a pretrained model from a local checkpoint. If False, then load file from
            model_dir directly

        """
        if model_dir is None:
            model_dir = self.model_dir
        self.model = BasicEnsembleModel(self.model_config)
        if from_no_finetune:
            self.model.from_default_no_finetune(model_dir)
        elif from_local_checkpoint:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            print(checkpoints)
            if len(checkpoints) == 0:
                raise ValueError("No checkpoint found")
            else:
                checkpoint = checkpoints[0]
                self.model.from_pretrained_local(checkpoint)

                data = torch.load(checkpoint, map_location=self.device)
                if data.get("global_step") is not None:
                    self._pretrained_global_step = data.get("global_step")
        else:
            self.model.from_pretrained_local(model_dir)

        self.model.to(self.device)

    def evaluate(self, test_dataset: Dataset, metric_string_list: list[str]):
        result = {}
        for metric in metric_string_list:
            if metric == "rms":
                result.update({"rms_score": self.evaluate_rms(test_dataset)})
            else:
                raise ValueError(f"Unrecognised metric {metric}")

        return result

    def evaluate_rms(self, test_dataset: Dataset):
        generator = self.default_generator(
            test_dataset,
            epochs=1,
            deterministic=True,
        )

        cumulative_squared_error = 0

        total_batch_size = 0

        with torch.no_grad():
            for batch in generator:
                inputs: OneOrMany[torch.Tensor]
                inputs, labels, weights = self._prepare_batch(batch)

                output: UncertaintyRegressionPredictionOutput = (
                    self.model.predict_mean_and_std(**inputs, compute_std=False)
                )

                batch_size = len(labels)

                cumulative_squared_error += torch.sum(
                    torch.pow(labels.squeeze() - output.mean, 2)
                )

                total_batch_size += batch_size

            return (
                (torch.sqrt(cumulative_squared_error / total_batch_size))
                .cpu()
                .detach()
                .numpy()
            )

    def fit(
        self,
        dataset: Dataset,
        nb_epoch: int = 10,
        max_checkpoints_to_keep: int = 5,
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
        max_test_batches: int = None,
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
        self._train_losses = []
        self._test_checkpoint_losses = []

        for member_name in self.model.member_names:
            logger.info(f"Training: {member_name}:")
            self.fit_generator(
                active_member_name=member_name,
                generator=self.default_generator(
                    dataset, epochs=nb_epoch, deterministic=deterministic
                ),
                max_checkpoints_to_keep=max_checkpoints_to_keep,
                checkpoint_interval=checkpoint_interval,
                callbacks=callbacks,
                all_losses=all_losses,
                test_log_frequency=test_log_frequency,
                is_epoch_test_logging=is_epoch_test_logging,
                test_dataset=test_dataset,
                max_test_batches=max_test_batches,
                early_stopping=early_stopping,
            )

        if self.wandb_logger is not None:
            self.wandb_logger.wandb_run.log(
                {
                    "combined train losses": wandb.Table(
                        columns=["member step", "train loss", "member"],
                        data=self._train_losses,
                    )
                }
            )
            self.wandb_logger.wandb_run.log(
                {
                    "combined test checkpoint losses": wandb.Table(
                        columns=["member step", "test checkpoint loss", "member"],
                        data=self._test_checkpoint_losses,
                    )
                }
            )
        return

    def fit_generator(
        self,
        active_member_name: str,
        generator: Iterable[Tuple[Any, Any, Any]],
        max_checkpoints_to_keep: int = 5,
        checkpoint_interval: int = 1000,
        callbacks: Union[Callable, List[Callable]] = [],
        all_losses: Optional[List[float]] = None,
        test_log_frequency: int = None,
        is_epoch_test_logging: bool = False,
        test_dataset: Dataset = None,
        max_test_batches: int = None,
        early_stopping: bool = True,
    ) -> float:
        """Train this model on data from a generator.

        Parameters
        ----------
        active_member_name: str
            the name of the member that is being trained. In the form "member_i"
        generator: generator
            this should generate batches, each represented as a tuple of the form
            (inputs, labels, weights).
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        checkpoint_interval: int
            the frequency at which to write checkpoints, measured in training steps.
            Set this to 0 to disable automatic checkpointing.
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
        self._ensure_built_lora_ensemble()
        self.model.train()
        self.model.set_active_member(active_member_name)
        if self.wandb_logger is not None:
            self.create_wandb_metric(active_member_name)
        avg_loss = 0.0
        last_avg_loss = 0.0
        averaged_batches = 0
        optimizer, lr_schedule = self._create_optimizer_and_lr_scheduler()
        time1 = time.time()

        self._check_test_dataset_specified(
            test_dataset, is_epoch_test_logging, test_log_frequency, checkpoint_interval
        )

        epoch_frequency = self._train_size // self.batch_size

        early_stopper = EarlyStopping(max_checkpoints_to_keep, early_stopping)

        # Main training loop.

        for batch in generator:
            inputs: OneOrMany[torch.Tensor]
            inputs, labels, weights = self._prepare_batch(batch)

            optimizer.zero_grad()
            outputs = self.model(**inputs)

            if self._loss_outputs is not None:
                outputs = [outputs[i] for i in self._loss_outputs]
            batch_loss = outputs.get("loss")
            batch_loss.backward()
            optimizer.step()
            if lr_schedule is not None:
                lr_schedule.step()
            self._global_step += 1
            self._member_steps[active_member_name] += 1

            current_step = self._member_steps[active_member_name]

            avg_loss += batch_loss

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
                if early_stopper.should_early_stop(test_avg_loss):
                    break
                else:
                    self.save_checkpoint(max_checkpoints_to_keep)
            for c in callbacks:
                c(self, current_step)
            if self.tensorboard and should_log:
                self._log_scalar_to_tensorboard("loss", batch_loss, current_step)
            if (self.wandb_logger is not None) and should_log:
                self._log_on_train(batch_loss, is_epoch_evaluation=False)

            if test_log_frequency is None:
                should_log_test = False
            else:
                should_log_test = current_step % test_log_frequency == 0
            if should_log_test:
                self._evaluate_and_log_on_test(test_dataset, max_test_batches)

            should_log_at_epoch = current_step % epoch_frequency == 0
            if should_log_at_epoch:
                if is_epoch_test_logging:
                    self._evaluate_and_log_on_test(
                        test_dataset, max_test_batches, is_epoch_evaluation=True
                    )
                if self.wandb_logger is not None:
                    self._log_on_train(batch_loss, is_epoch_evaluation=True)

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
            if early_stopper.stopped:
                self.restore(checkpoint_number=early_stopper.optimal_checkpoint)
                print(
                    f"Early stopping {self.model.active_member} at {current_step} steps"
                )
            else:
                # perform check because final step may not be at checkpoint interval
                final_loss = self._evaluate_and_log_on_test(
                    test_dataset, is_checkpoint_evaluation=True
                )
                if early_stopper.should_early_stop(final_loss):
                    self.restore(checkpoint_number=early_stopper.optimal_checkpoint)
                else:
                    self.save_checkpoint(max_checkpoints_to_keep)
                    self.restore(checkpoint_number=early_stopper.optimal_checkpoint)
            self.save_checkpoint(max_checkpoints_to_keep + 1)

        time2 = time.time()
        logger.info("TIMING: model fitting took %0.3f s" % (time2 - time1))
        return last_avg_loss

    def _ensure_built_lora_ensemble(self):
        self._member_steps = {member: 0 for member in self.model.member_names}

        # remove files in checkpoint directory
        model_dir = self.model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for f in os.listdir(model_dir):
            os.remove(os.path.join(model_dir, f))

    def _create_optimizer_and_lr_scheduler(self):
        active_param_list = []

        for param in self.model.parameters():
            if param.requires_grad == True:
                active_param_list.append(param)

        optimizer = self.optimizer._create_pytorch_optimizer(active_param_list)

        if isinstance(self.optimizer.learning_rate, LearningRateSchedule):
            lr_schedule = self.optimizer.learning_rate._create_pytorch_schedule(
                optimizer
            )
        else:
            lr_schedule = None

        return optimizer, lr_schedule

    def create_wandb_metric(self, active_member_name: str):
        self.wandb_logger.wandb_run.define_metric(f"{active_member_name} step")

        self.wandb_logger.wandb_run.define_metric(
            f"train/{active_member_name} loss",
            step_metric=f"{active_member_name} step",
        )
        self.wandb_logger.wandb_run.define_metric(
            f"train/{active_member_name} epoch loss",
            step_metric=f"{active_member_name} step",
        )

        self.wandb_logger.wandb_run.define_metric(
            f"test/{active_member_name} loss",
            step_metric=f"{active_member_name} step",
        )
        self.wandb_logger.wandb_run.define_metric(
            f"test/{active_member_name} epoch loss",
            step_metric=f"{active_member_name} step",
        )
        self.wandb_logger.wandb_run.define_metric(
            f"test/{active_member_name} checkpoint loss",
            step_metric=f"{active_member_name} step",
        )

    def _log_on_train(
        self,
        batch_loss,
        is_epoch_evaluation: bool = False,
    ):
        active_member_name = self.model.active_member

        if is_epoch_evaluation:
            evalutation_type_str = "epoch loss"
        else:
            evalutation_type_str = "loss"
            self._train_losses.append(
                [
                    self._member_steps[active_member_name],
                    batch_loss,
                    active_member_name,
                ]
            )

        all_data = dict(
            {
                f"{active_member_name} step": self._member_steps[active_member_name],
                f"train/{active_member_name} {evalutation_type_str}": batch_loss,
            }
        )

        self.wandb_logger.wandb_run.log(
            all_data,
        )

    def _evaluate_and_log_on_test(
        self,
        test_dataset: Dataset,
        max_test_batches: int | None = None,
        is_epoch_evaluation: bool = False,
        is_checkpoint_evaluation: bool = False,
        log_output: bool = True,
    ):
        self.model.eval()
        sum_avg_loss = 0.0
        averaged_batches = 0
        active_member_name = self.model.active_member

        generator = self.default_generator(
            dataset=test_dataset,
            epochs=1,
            mode="predict",
            deterministic=False,
            pad_batches=False,
        )

        with torch.no_grad():
            for batch in generator:
                inputs: OneOrMany[torch.Tensor]
                inputs, _, _ = self._prepare_batch(batch)
                output_values = self.model(**inputs)
                batch_loss = output_values.get("loss")

                sum_avg_loss += batch_loss
                averaged_batches += 1

                if max_test_batches is not None:
                    if averaged_batches >= max_test_batches:
                        break

            avg_loss = float(sum_avg_loss) / averaged_batches

            if (self.wandb_logger is not None) and (log_output == True):
                if is_epoch_evaluation:
                    evalutation_type_str = "epoch loss"
                elif is_checkpoint_evaluation:
                    evalutation_type_str = "checkpoint loss"
                    self._test_checkpoint_losses.append(
                        [
                            self._member_steps[active_member_name],
                            avg_loss,
                            active_member_name,
                        ]
                    )
                else:
                    evalutation_type_str = "loss"

                all_data = dict(
                    {
                        f"{active_member_name} step": self._member_steps[
                            active_member_name
                        ],
                        f"test/{active_member_name} {evalutation_type_str}": avg_loss,
                    }
                )

                self.wandb_logger.wandb_run.log(all_data)

        return avg_loss

    def restore(
        self,
        checkpoint: Optional[str] = None,
        checkpoint_number: int = 1,
        model_dir: Optional[str] = None,
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
        self.model.load_state_dict(data["model_state_dict"])
        self.model.set_active_member(data["active_member"])
        self._pytorch_optimizer.load_state_dict(
            data["optimizer_state_dict"]
        )  # redundant
        self._global_step = data["global_step"]
        self._member_steps = data["member_steps"]

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
            "member_steps": self._member_steps,
            "active_member": self.model.active_member,
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
