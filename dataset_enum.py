import numpy as np
import enum
import itertools
import ignite
import torch
from torch import optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchvision import datasets, transforms

from models import cifar_model, mnist_model, emnist_model
from active_learning_data import ActiveLearningData
from torch_utils import get_balanced_sample_indices, balance_dataset_by_repeating, get_target_bins, get_targets
from transformed_dataset import TransformedDataset
import subrange_dataset
from ignite_progress_bar import ignite_progress_bar
import ignite_restoring_score_guard
from ignite_utils import epoch_chain, chain, log_epoch_results, store_epoch_results, store_iteration_results

from sampler_model import SamplerModel, NoDropoutModel
from random_fixed_length_sampler import RandomFixedLengthSampler

from ignite.exceptions import NotComputableError


# from sampler_model import SamplerModel, NoDropoutModel

# IDK what the heck this does

def filter_dataset(dataset, class_select=[0, 3, 5, 7], keep_prob=0.3):
    keep_prob_dict = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
    for class_temp in class_select:
        keep_prob_dict[class_temp] = keep_prob

    filter = np.ones(len(dataset), dtype=np.bool)
    for class_down in keep_prob_dict:
        filter_class = np.array(dataset.targets) == class_down
        filter_random = np.random.rand(len(dataset)) > keep_prob_dict[class_down]
        filter = np.logical_and(~np.logical_and(filter_class, filter_random), filter)
    dataset.data = dataset.data[filter]
    dataset.targets = np.array(dataset.targets)[filter].tolist()


class EMNIST(datasets.EMNIST):
    url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'

    def __init__(self, root, split, **kwargs):
        super(EMNIST, self).__init__(root, split, **kwargs)


class Entropy(ignite.metrics.Metric):
    """
    Calculates the entropy of the prediction.

    - `update` must receive output of the form `(y_pred, y)`.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self._sum_of_entropy = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        entropy = torch.sum(-torch.exp(y_pred) * y_pred, dim=1)
        self._sum_of_entropy += torch.sum(entropy).item()
        self._num_examples += y_pred.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("Entropy must have at"
                                     "least one example before it can be computed.")
        return self._sum_of_entropy / self._num_examples


def build_metrics():
    return {"accuracy": ignite.metrics.Accuracy(), "nll": ignite.metrics.Loss(F.nll_loss), "entropy": Entropy()}


def compose_transformers(iterable):
    iterable = list(filter(None, iterable))
    if len(iterable) == 0:
        return None
    if len(iterable) == 1:
        return iterable[0]
    return transforms.Compose(iterable)


class DatasetEnum(enum.Enum):
    mnist = "mnist"
    mnist_w_noise = "mnist_w_noise"
    repeated_mnist = "repeated_mnist"
    repeated_mnist_w_noise = "repeated_mnist_w_noise"
    emnist = "emnist"
    emnist_bymerge = "emnist_bymerge"
    fmnist = 'fmnist'
    cifar = "cifar"
    cifar_unbalanced = "cifar_unbalanced"
    cifar100 = "cifar100"

    def get_data_source(self) -> None:
        self.train_dataset: Dataset = None
        self.available_dataset: Dataset = None
        self.validation_dataset: Dataset = None
        self.test_dataset: Dataset = None
        self.shared_transform: object = None
        self.train_transform: object = None
        self.scoring_transform: object = None
        self.active_learning_data = None
        self.initial_samples = None

        # MNIST
        if self in (DatasetEnum.mnist, DatasetEnum.mnist_w_noise,
                    DatasetEnum.repeated_mnist, DatasetEnum.repeated_mnist_w_noise):
            # num_classes = 10, input_size = 28
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            self.train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
            self.test_dataset = datasets.MNIST("data", train=False, transform=transform)

            if self == DatasetEnum.mnist:
                return
            elif self in (DatasetEnum.repeated_mnist, DatasetEnum.repeated_mnist_w_noise):
                self.train_dataset = data.ConcatDataset([self.train_dataset] * 3)
            if self in (DatasetEnum.mnist_w_noise, DatasetEnum.repeated_mnist_w_noise):
                dataset_noise = torch.empty(
                    (len(self.train_dataset), 28, 28), dtype=torch.float32
                ).normal_(0.0, 0.1)

                def apply_noise(idx, sample):
                    _data, _target = sample
                    return _data + dataset_noise[idx], _target

                self.train_dataset = TransformedDataset(self.train_dataset, transformer=apply_noise)
            return

        # FMNIST
        elif self == DatasetEnum.fmnist:
            transform = transforms.ToTensor()
            self.train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
            self.test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
            return

        # EMNIST
        elif self in (DatasetEnum.emnist, DatasetEnum.emnist_bymerge):
            # num_classes = 47, input_size = 28
            split = "balanced" if self == DatasetEnum.emnist else "bymerge"
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            self.train_dataset = EMNIST("emnist_data", split=split, train=True, download=True, transform=transform)
            self.test_dataset = EMNIST("emnist_data", split=split, train=False, transform=transform)

            """
                Table II contains a summary of the EMNIST datasets and
                indicates which classes contain a validation subset in the
                training set. In these datasets, the last portion of the training
                set, equal in size to the testing set, is set aside as a validation
                set. Additionally, this subset is also balanced such that it
                contains an equal number of samples for each task. If the
                validation set is not to be used, then the training set can be
                used as one contiguous set.
            """
            if self == DatasetEnum.emnist:
                # Balanced contains a test set
                split_index = len(self.train_dataset) - len(self.test_dataset)
                self.train_dataset, self.validation_dataset = subrange_dataset.dataset_subset_split(self.train_dataset,
                                                                                                    split_index)
            else:
                self.validation_dataset = None
            return

        # CIFAR
        elif self in (DatasetEnum.cifar, DatasetEnum.cifar_unbalanced):
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            self.test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            if self == DatasetEnum.cifar_unbalanced:
                filter_dataset(self.train_dataset)
                # filter_dataset(test_dataset)
            return

        # CIFAR100
        elif self == DatasetEnum.cifar100:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
            self.test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
            return

        raise NotImplementedError(f"Unknown dataset {self}!")

    @property
    def num_classes(self):
        if self in (DatasetEnum.mnist, DatasetEnum.repeated_mnist_w_noise, DatasetEnum.mnist_w_noise,
                    DatasetEnum.fmnist, DatasetEnum.cifar, DatasetEnum.cifar_unbalanced):
            return 10
        elif self in (DatasetEnum.emnist, DatasetEnum.emnist_bymerge):
            return 47
        elif self == DatasetEnum.cifar100:
            return 100
        raise NotImplementedError(f"Unknown dataset {self}!")

    def create_bayesian_model(self, device):
        num_classes = self.num_classes
        self.model = None
        self._modelCreated = True
        if self in (DatasetEnum.mnist, DatasetEnum.repeated_mnist_w_noise,
                    DatasetEnum.mnist_w_noise, DatasetEnum.fmnist):
            self.model = mnist_model.BayesianNet(num_classes=num_classes).to(device)
        elif self in (DatasetEnum.emnist, DatasetEnum.emnist_bymerge):
            self.model = emnist_model.BayesianNet(num_classes=num_classes).to(device)
        elif self in (DatasetEnum.cifar, DatasetEnum.cifar_unbalanced, DatasetEnum.cifar100):
            self.model = cifar_model.BayesianNet(num_classes=num_classes).to(device)
        else:
            self._modelCreated = False
            self.model = None
            raise NotImplementedError(f"Unknown dataset {self}!")

    def create_optimizer(self):
        self._optimizerCreated = False
        if self._modelCreated:
            self._optimizerCreated = True
            self.optimizer = optim.Adam(self.model.parameters())  # ,lr=1e-2)
        # optimizer1 = optim.SGD(model.parameters(),lr=0.1, momentum=0.9, weight_decay=5e-4)

    def create_scheduler(self):
        self._schedulerCreated = False
        if self._optimizerCreated:
            self._schedulerCreated = True
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=0, last_epoch=-1)

    def get_experiment_data(self, initial_samples,
                            reduced_dataset,
                            samples_per_class,
                            validation_set_size,
                            balanced_test_set,
                            balanced_validation_set):
        num_classes = self.num_classes
        self.active_learning_data = ActiveLearningData(self.train_dataset)
        if not initial_samples:
            initial_samples = list(
                itertools.chain.from_iterable(
                    get_balanced_sample_indices(
                        get_targets(self.train_dataset), num_classes=num_classes, n_per_digit=samples_per_class
                    ).values()
                )
            )

        self.active_learning_data.acquire(initial_samples)

        # Split off the validation dataset after acquiring the initial samples.
        if not self.validation_dataset:
            print("Acquiring validation set from training set.")
            if not validation_set_size:
                validation_set_size = len(self.test_dataset)

            if not balanced_validation_set:
                self.validation_dataset = self.active_learning_data.extract_dataset(validation_set_size)
            else:
                print("Using a balanced validation set")
                self.validation_dataset = self.active_learning_data.extract_dataset_from_indices(
                    balance_dataset_by_repeating(
                        self.active_learning_data.available_dataset, num_classes, validation_set_size, upsample=False
                    )
                )
        else:
            if validation_set_size == 0:
                print("Using provided validation set.")
                validation_set_size = len(self.validation_dataset)
            if validation_set_size < len(self.validation_dataset):
                print("Shrinking provided validation set.")
                if not balanced_validation_set:
                    self.validation_dataset = data.Subset(
                        self.validation_dataset,
                        torch.randperm(len(self.validation_dataset))[:validation_set_size].tolist()
                    )
                else:
                    print("Using a balanced validation set")
                    self.validation_dataset = data.Subset(
                        self.validation_dataset,
                        balance_dataset_by_repeating(self.validation_dataset, num_classes, validation_set_size),
                    )

        if balanced_test_set:
            print("Using a balanced test set")
            print("Distribution of original test set classes:")
            classes = get_target_bins(self.test_dataset)
            print(classes)

            self.test_dataset = data.Subset(
                self.test_dataset, balance_dataset_by_repeating(self.test_dataset, num_classes, len(self.test_dataset))
            )

        if reduced_dataset:
            print("USING REDUCED DATASET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Let's assume we won't use more than 1000 elements for our validation set.
            self.active_learning_data.extract_dataset(
                len(self.train_dataset) - max(len(self.train_dataset) // 20, 5000))
            test_dataset = subrange_dataset.SubrangeDataset(self.test_dataset, 0,
                                                            max(len(self.test_dataset) // 10, 5000))
            if self.validation_dataset:
                self.validation_dataset = subrange_dataset.SubrangeDataset(self.validation_dataset, 0,
                                                                           len(self.validation_dataset) // 10)

        show_class_frequencies = True
        if show_class_frequencies:
            print("Distribution of training set classes:")
            print(get_target_bins(self.train_dataset))

            print("Distribution of validation set classes:")
            print(get_target_bins(self.validation_dataset))

            print("Distribution of test set classes:")
            print(get_target_bins(self.test_dataset))

            print("Distribution of pool classes:")
            print(get_target_bins(self.active_learning_data.available_dataset))

            print("Distribution of active set classes:")
            print(get_target_bins(self.active_learning_data.active_dataset))

        print(f"Dataset info:")
        print(f"\t{len(self.active_learning_data.active_dataset)} active samples")
        print(f"\t{len(self.active_learning_data.available_dataset)} available samples")
        print(f"\t{len(self.validation_dataset)} validation samples")
        print(f"\t{len(self.test_dataset)} test samples")

        if self.shared_transform or self.train_transform:
            self.train_dataset = TransformedDataset(
                self.active_learning_data.active_dataset,
                vision_transformer=compose_transformers([self.train_transform, self.shared_transform]),
            )
        else:
            self.train_dataset = self.active_learning_data.active_dataset

        if self.shared_transform or self.scoring_transform:
            self.available_dataset = TransformedDataset(
                self.active_learning_data.available_dataset,
                vision_transformer=compose_transformers([self.scoring_transform, self.shared_transform]),
            )
        else:
            self.available_dataset = self.active_learning_data.available_dataset

        if self.shared_transform:
            self.test_dataset = TransformedDataset(self.test_dataset, vision_transformer=self.shared_transform)
            self.validation_dataset = TransformedDataset(self.validation_dataset,
                                                         vision_transformer=self.shared_transform)

        self.initial_samples = initial_samples

    def prepare_data_loaders(self, epoch_samples, train_batch_size, test_batch_size, scoring_batch_size, **kwargs):
        self.train_loader = DataLoader(self.train_dataset,
                                       sampler=RandomFixedLengthSampler(self.train_dataset, epoch_samples),
                                       batch_size=train_batch_size, **kwargs)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=test_batch_size, shuffle=False,
                                            **kwargs)
        self.test_loader = DataLoader(self.test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
        self.available_loader = DataLoader(self.available_dataset, batch_size=scoring_batch_size, shuffle=False,
                                           **kwargs)

    def train_model(
            self,
            num_inference_samples,
            max_epochs,
            early_stopping_patience,
            desc,
            log_interval,
            device,
            epoch_results_store=None,
    ):
        self.create_bayesian_model(device)
        self.create_optimizer()
        self.create_scheduler()
        num_lr_epochs = 0

        training_sampler = SamplerModel(self.model, k=1).to(device)
        test_sampler = SamplerModel(self.model, k=min(num_inference_samples, 100)).to(device)
        validation_sampler = NoDropoutModel(self.model).to(device)

        trainer = ignite.engine.create_supervised_trainer(training_sampler, self.optimizer, F.nll_loss, device=device)
        validator = ignite.engine.create_supervised_evaluator(validation_sampler,
                                                              metrics=build_metrics(),
                                                              device=device)

        def out_of_patience():
            nonlocal num_lr_epochs
            if num_lr_epochs <= 0 or self.scheduler is None:
                trainer.terminate()
            else:
                self.scheduler.step()
                restoring_score_guard.patience = int(restoring_score_guard.patience * 1.5 + 0.5)
                print(f"New LRs: {[group['lr'] for group in self.optimizer.param_groups]}")
                num_lr_epochs -= 1

        if self.scheduler:
            print(f"LRs: {[group['lr'] for group in self.optimizer.param_groups]}")

        # Cut training if there are no improvements
        restoring_score_guard = ignite_restoring_score_guard.RestoringScoreGuard(
            patience=early_stopping_patience,
            score_function=lambda engine: engine.state.metrics["accuracy"],
            out_of_patience_callback=out_of_patience,
            module=self.model,
            optimizer=self.optimizer,
            training_engine=trainer,
            validation_engine=validator,
        )

        if self.test_loader:
            tester = ignite.engine.create_supervised_evaluator(test_sampler, metrics=build_metrics(),
                                                               device=device)
            ignite_progress_bar(tester, desc("Test Eval"), log_interval)
            # run tester when training is complete
            chain(trainer, tester, self.test_loader)
            log_epoch_results(tester, "Test", trainer)

        ignite_progress_bar(trainer, desc("Training"), log_interval)
        ignite_progress_bar(validator, desc("Validation Eval"), log_interval)

        # NOTE(blackhc): don't run a full test eval after every epoch.
        # epoch_chain(trainer, test_evaluator, test_loader)

        # run validator when each epoch is complete
        epoch_chain(trainer, validator, self.validation_loader)

        log_epoch_results(validator, "Validation", trainer)

        if epoch_results_store is not None:
            epoch_results_store["validations"] = []
            epoch_results_store["losses"] = []
            store_epoch_results(validator, epoch_results_store["validations"])
            store_iteration_results(trainer, epoch_results_store["losses"], log_interval=2)

            if self.test_loader is not None:
                store_epoch_results(tester, epoch_results_store, name="test")

        if len(self.train_loader.dataset) > 0:
            trainer.run(self.train_loader, max_epochs)
        else:
            tester.run(self.test_loader)

        num_epochs = trainer.state.epoch if trainer.state else 0

        test_metrics = None
        if self.test_loader is not None:
            test_metrics = tester.state.metrics

        return self.model, num_epochs, test_metrics
