from blackhc.progress_bar import with_progress_bar
from torch import nn as nn
from acquisition.functions import AcquisitionFunction
from torch.utils import data
import active_learning_data

from models import mc_dropout
import torch

import torch_utils

eval_bayesian_model_consistent_cuda_chunk_size = 1024
sampler_model_cuda_chunk_size = 1024


def eval_bayesian_model_consistent(
    bayesian_model: mc_dropout.BayesianModule, available_loader, num_classes, k=20, device=None
):
    global eval_bayesian_model_consistent_cuda_chunk_size

    with torch.no_grad():
        # NOTE: I'm hard-coding 10 classes here!
        B = len(available_loader.dataset)
        logits_B_K_C = torch.empty((B, k, num_classes), dtype=torch.float64)

        chunk_size = eval_bayesian_model_consistent_cuda_chunk_size if device.type == "cuda" else 64
        torch_utils.gc_cuda()
        k_lower = 0
        while k_lower < k:
            try:
                k_upper = min(k_lower + chunk_size, k)

                # This resets the dropout masks.
                bayesian_model.eval()

                for i, (batch, _) in enumerate(
                    with_progress_bar(available_loader, unit_scale=available_loader.batch_size)
                ):
                    lower = i * available_loader.batch_size
                    upper = min(lower + available_loader.batch_size, B)

                    batch = batch.to(device)
                    # batch_size x ws x classes
                    mc_output_B_K_C = bayesian_model(batch, k_upper - k_lower)
                    logits_B_K_C[lower:upper, k_lower:k_upper].copy_(mc_output_B_K_C.double(), non_blocking=True)

            except RuntimeError as exception:
                if torch_utils.should_reduce_batch_size(exception):
                    if chunk_size <= 1:
                        raise
                    chunk_size //= 2
                    print(f"New eval_bayesian_model_consistent_cuda_chunk_size={chunk_size} ({exception})")
                    eval_bayesian_model_consistent_cuda_chunk_size = chunk_size

                    torch_utils.gc_cuda()
                else:
                    raise
            else:
                k_lower += chunk_size

    return logits_B_K_C


reduced_eval_consistent_bayesian_model_cuda_chunk_size = 512


def reduced_eval_consistent_bayesian_model(
    bayesian_model: mc_dropout.BayesianModule,
    acquisition_function: AcquisitionFunction,
    num_classes: int,
    k: int,
    initial_percentage: int,
    reduce_percentage: int,
    target_size: int,
    available_loader,
    device=None,
) -> (active_learning_data.ActiveLearningData, data.DataLoader, torch.Tensor, torch.Tensor):
    """Performs a scoring step with k inference samples while reducing the dataset to at most min_remaining_percentage.

    Before computing anything at all the initial available dataset is randomly culled to initial_percentage.

    Every `chunk_size` inferences BALD is recomputed and the bottom `reduce_percentage` samples are dropped."""
    global reduced_eval_consistent_bayesian_model_cuda_chunk_size

    # Here, we need to use available_dataset because it allows us to easily recover the original indices.

    # We start with all data in the acquired data.
    subset_split = active_learning_data.ActiveLearningData(available_loader.dataset)
    initial_length = len(available_loader.dataset)

    initial_split_length = initial_length * initial_percentage // 100

    # By acquiring [initial_split_length:], we make the tail unavailable.
    subset_split.acquire(torch.randperm(initial_length)[initial_split_length:])

    subset_dataloader = data.DataLoader(
        subset_split.available_dataset, shuffle=False, batch_size=available_loader.batch_size
    )

    print(f"Scoring subset of {len(subset_dataloader.dataset)} items:")

    with torch.no_grad():
        B = len(subset_split.available_dataset)
        C = num_classes

        # We stay on the CPU.
        logits_B_K_C = None

        k_lower = 0
        torch_utils.gc_cuda()
        chunk_size = reduced_eval_consistent_bayesian_model_cuda_chunk_size if device.type == "cuda" else 32
        while k_lower < k:
            try:
                k_upper = min(k_lower + chunk_size, k)

                old_logit_B_K_C = logits_B_K_C
                # This also stays on the CPU.
                logits_B_K_C = torch.empty((B, k_upper, C), dtype=torch.float64)

                # Copy the old data over.
                if k_lower > 0:
                    logits_B_K_C[:, 0:k_lower, :].copy_(old_logit_B_K_C)

                # This resets the dropout masks.
                bayesian_model.eval()

                for i, (batch, _) in enumerate(
                    with_progress_bar(subset_dataloader, unit_scale=subset_dataloader.batch_size)
                ):
                    lower = i * subset_dataloader.batch_size
                    upper = min(lower + subset_dataloader.batch_size, B)

                    batch = batch.to(device)
                    # batch_size x ws x classes
                    mc_output_B_K_C = bayesian_model(batch, k_upper - k_lower)
                    logits_B_K_C[lower:upper, k_lower:k_upper].copy_(mc_output_B_K_C.double(), non_blocking=True)

            except RuntimeError as exception:
                if torch_utils.should_reduce_batch_size(exception):
                    if chunk_size <= 1:
                        raise
                    chunk_size = chunk_size // 2
                    print(f"New reduced_eval_consistent_bayesian_model_cuda_chunk_size={chunk_size} ({exception})")
                    reduced_eval_consistent_bayesian_model_cuda_chunk_size = chunk_size

                    torch_utils.gc_cuda()
                else:
                    raise
            else:
                if k_upper == k:
                    next_size = target_size
                elif k_upper < 50:
                    next_size = B
                else:
                    next_size = max(target_size, B * (100 - reduce_percentage) // 100)

                # Compute the score if it's needed: we are going to reduce the dataset or we're in the last iteration.
                if next_size < B or k_upper == k:
                    scores_B = acquisition_function.compute_scores(
                        logits_B_K_C, available_loader=subset_dataloader, device=device
                    )
                else:
                    scores_B = None

                if next_size < B:
                    print("Reducing size", next_size)
                    sorted_indices = torch.argsort(scores_B, descending=True)

                    new_indices = torch.sort(sorted_indices[:next_size], descending=False)[0]

                    B = next_size
                    logits_B_K_C = logits_B_K_C[new_indices]
                    if k_upper == k:
                        logits_B_K_C = logits_B_K_C.clone().detach()
                    scores_B = scores_B[new_indices].clone().detach()

                    # Acquire all the low scorers
                    subset_split.acquire(sorted_indices[next_size:])

                k_lower += chunk_size

    return subset_split, subset_dataloader, scores_B, logits_B_K_C


class NoDropoutModel(nn.Module):
    def __init__(self, bayesian_net: mc_dropout.BayesianModule):
        super().__init__()
        self.bayesian_net = bayesian_net

    def forward(self, input):
        self.bayesian_net.set_dropout_p(0)
        mc_output_B_1_C = self.bayesian_net(input, 1)
        self.bayesian_net.set_dropout_p(mc_dropout.DROPOUT_PROB)
        return mc_output_B_1_C.squeeze(1)


class SamplerModel(nn.Module):
    def __init__(self, bayesian_net: mc_dropout.BayesianModule, k):
        super().__init__()
        self.bayesian_net = bayesian_net
        self.num_classes = bayesian_net.num_classes
        self.k = k

    def forward(self, input: torch.Tensor):
        global sampler_model_cuda_chunk_size
        if self.training:
            return torch_utils.logit_mean(self.bayesian_net(input, self.k), dim=1, keepdim=False)
        else:
            mc_output_B_C = torch.zeros((input.shape[0], self.num_classes), dtype=torch.float64, device=input.device)

            k = self.k

            chunk_size = sampler_model_cuda_chunk_size if input.device.type == "cuda" else 32

            k_lower = 0
            while k_lower < k:
                try:
                    k_upper = min(k_lower + chunk_size, k)

                    # Reset the mask all around.
                    self.bayesian_net.eval()

                    mc_output_B_K_C = self.bayesian_net(input, k_upper - k_lower)
                except RuntimeError as exception:
                    if torch_utils.should_reduce_batch_size(exception):
                        chunk_size //= 2
                        if chunk_size <= 0:
                            raise
                        if sampler_model_cuda_chunk_size != chunk_size:
                            print(f"New sampler_model_cuda_chunk_size={chunk_size} ({exception})")
                            sampler_model_cuda_chunk_size = chunk_size

                        torch_utils.gc_cuda()
                else:
                    mc_output_B_C += torch.sum(mc_output_B_K_C.double().exp_(), dim=1, keepdim=False)
                    k_lower += chunk_size

            return (mc_output_B_C / k).log_()
