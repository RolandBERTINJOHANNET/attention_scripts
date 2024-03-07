import logging
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)

from typing import Mapping

from tqdm import tqdm

from lightning.pytorch.loggers.wandb import WandbLogger
from shimmer import ContrastiveLossType
from shimmer.modules.global_workspace import (
    GlobalWorkspace,
    GlobalWorkspaceFusion,
    SchedulerArgs,
    VariationalGlobalWorkspace,
)
from torch import set_float32_matmul_precision

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.ckpt_migrations import (
    SaveMigrations,
    gw_migrations,
    var_gw_migrations,
)
from simple_shapes_dataset.config import load_config
from simple_shapes_dataset.dataset import SimpleShapesDataModule
from simple_shapes_dataset.dataset.pre_process import (
    color_blind_visual_domain,
    nullify_attribute_rotation,
)
from simple_shapes_dataset.logging import LogGWImagesCallback
from simple_shapes_dataset.modules.contrastive_loss import VSEPPContrastiveLoss
from simple_shapes_dataset.modules.domains import load_pretrained_domains

device = torch.device("cuda:0")

def to_device(batch, device):
    """Recursively move the batch to the specified device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    else:
        return batch

def scalar_multiply(batch, scalar):
    """Recursively move the batch to the specified device."""
    if isinstance(batch, torch.Tensor):
        return batch*scalar
    elif isinstance(batch, dict):
        return {k: to_device(v, scalar) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [to_device(v, scalar) for v in batch]
    else:
        print("failed!!!!!!!!!!!")
        exit()
        return batch

def sample_scaling_factor(a, temperature=1000):
    """
    Samples a scaling factor k from a uniform distribution and applies softmax with a specified temperature.
    Returns the scaling factors along with the method used ('binary' or 'softmax').
    """
    if torch.rand(1).item() < a:
        # Sample binary (0 or 1)
        k = torch.randint(0, 2, (1,)).float()
        softmax_values = torch.tensor([k, 1 - k])
        method = 'binary'
    else:
        # Sample uniformly from the entire range [0, 1]
        k = torch.rand(1)
        # Apply softmax with temperature
        softmax_values = F.softmax(torch.tensor([k, 1 - k]) * temperature, dim=0)
        method = 'softmax'
    return softmax_values[0].item(), softmax_values[1].item(), method


config = load_config(
    "../config",
    load_files=["train_gw.yaml"],
    debug_mode=DEBUG_MODE,
)

import random

# Generate a random integer between 0 and 99999 (you can choose your range)
random_seed = random.randint(0, 99999)
print("seed : ",random_seed)


# Use the generated random seed
seed_everything(random_seed, workers=True)

domain_proportion = {
    frozenset(item.domains): item.proportion
    for item in config.global_workspace.domain_proportions
}

additional_transforms: dict[str, list[Callable[[Any], Any]]] = {}
if config.domain_modules.attribute.nullify_rotation:
    logging.info("Nullifying rotation in the attr domain.")
    additional_transforms["attr"] = [nullify_attribute_rotation]
if config.domain_modules.visual.color_blind:
    logging.info("v domain will be color blind.")
    additional_transforms["v"] = [color_blind_visual_domain]

data_module = SimpleShapesDataModule(
    config.dataset.path,
    domain_proportion,
    batch_size=config.training.batch_size,
    num_workers=config.training.num_workers,
    seed=config.seed,
    ood_seed=config.ood_seed,
    domain_args=config.global_workspace.domain_args,
    additional_transforms=additional_transforms,
)

domain_modules, interfaces = load_pretrained_domains(
    config.default_root_dir,
    config.global_workspace.domains,
    config.global_workspace.latent_dim,
    config.global_workspace.encoders.hidden_dim,
    config.global_workspace.encoders.n_layers,
    config.global_workspace.decoders.hidden_dim,
    config.global_workspace.decoders.n_layers,
    is_variational=config.global_workspace.is_variational,
    is_linear=config.global_workspace.linear_domains,
    bias=config.global_workspace.linear_domains_use_bias,
)

loss_coefs: dict[str, torch.Tensor] = {
    "demi_cycles": torch.Tensor(
        [config.global_workspace.loss_coefficients.demi_cycles]
    ),
    "cycles": torch.Tensor([config.global_workspace.loss_coefficients.cycles]),
    "translations": torch.Tensor(
        [config.global_workspace.loss_coefficients.translations]
    ),
    "contrastives": torch.Tensor(
        [config.global_workspace.loss_coefficients.contrastives]
    ),
}

# if config.global_workspace.is_variational:
#     loss_coefs["kl"] = torch.Tensor(
#         [config.global_workspace.loss_coefficients.kl]
#     )

print("using it : ",config.global_workspace.use_fusion_model)

contrastive_fn: ContrastiveLossType | None = None
if config.global_workspace.vsepp_contrastive_loss:
    contrastive_fn = VSEPPContrastiveLoss(
        config.global_workspace.vsepp_margin,
        config.global_workspace.vsepp_measure,
        config.global_workspace.vsepp_max_violation,
        torch.tensor([1 / 0.07]).log(),
    )

if config.global_workspace.is_variational:
    module = VariationalGlobalWorkspace(
        domain_modules,
        interfaces,
        config.global_workspace.latent_dim,
        loss_coefs,
        config.global_workspace.var_contrastive_loss,
        config.training.optim.lr,
        config.training.optim.weight_decay,
        scheduler_args=SchedulerArgs(
            max_lr=config.training.optim.max_lr,
            total_steps=config.training.max_steps,
        ),
        learn_logit_scale=config.global_workspace.learn_logit_scale,
        contrastive_loss=contrastive_fn,
    )
elif config.global_workspace.use_fusion_model:
    module = GlobalWorkspaceFusion(
        domain_modules,
        interfaces,
        config.global_workspace.latent_dim,
        loss_coefs,
        config.training.optim.lr,
        config.training.optim.weight_decay,
        scheduler_args=SchedulerArgs(
            max_lr=config.training.optim.max_lr,
            total_steps=config.training.max_steps,
        ),
        learn_logit_scale=config.global_workspace.learn_logit_scale,
        contrastive_loss=contrastive_fn,
    )
    print("here!")
else:
    print("here!")
    module = GlobalWorkspace(
        domain_modules,
        interfaces,
        config.global_workspace.latent_dim,
        loss_coefs,
        config.training.optim.lr,
        config.training.optim.weight_decay,
        scheduler_args=SchedulerArgs(
            max_lr=config.training.optim.max_lr,
            total_steps=config.training.max_steps,
        ),
        learn_logit_scale=config.global_workspace.learn_logit_scale,
        contrastive_loss=contrastive_fn,
    )


########################################################################   loading the pretrained fusion translators

        # Path to your checkpoint file
checkpoint_path = os.getenv('CHECKPOINT_PATH')

        # Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:0'))

        # Assuming module is your DeterministicGlobalWorkspace model
        # Load the state dict from the checkpoint into your model
module.load_state_dict(checkpoint['state_dict'])
module = module.to(torch.device("cuda:0"))

def to_device(batch, device):
    """Recursively move the batch to the specified device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    else:
        return batch

########################################################################   training loop



num_epochs=10

training_losses = []
validation_losses = []

data_module.setup(stage='fit')  # This sets up the training dataset


def prepare_data_for_training(batch,device):
    # Move batch to the specified device
    batch = to_device(batch, device)
    attr_data = None
    for domain_key, domain_data in batch.items():
        if 'attr' in domain_key:
            attr_data = domain_data['attr']
            break

    if attr_data is None or not isinstance(attr_data, list) or len(attr_data) == 0:
        return None, None

    shape_info = attr_data[0]  # The first tensor in the attr_data list contains the shape information
    return shape_info, batch

def compute_loss_and_backpropagate(shape_info, global_workspace_vector, criterion, optimizer):

    shape_class_indices = torch.argmax(shape_info, dim=1)  # Assuming one-hot encoding

    # Compute loss
    shape_logits = classification_head(global_workspace_vector)
    loss = criterion(shape_logits, shape_class_indices)  # Use class indices
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Generate 20 scaling factors between 0 and 1
scaling_factors = torch.linspace(0., 1., 20, device='cuda:0')

# Initialize lists to store the average norms for each scaling factor
avg_norms_v_latents = []
avg_norms_attr = []

for scaling_factor in scaling_factors:
    total_norm_v_latents = 0
    total_norm_attr = 0
    total_batches = 0

    with torch.no_grad():
        for batch_tuple in tqdm(iter(data_module.val_dataloader())):
            batch = batch_tuple[0]
            adjusted_batch = {}
            for key, value in batch.items():
                adjusted_batch[frozenset([key])] = {key: value}

            shape_info, adjusted_batch = prepare_data_for_training(adjusted_batch, device)
            if shape_info is None:
                continue  # Skip the batch if shape information is missing

            latents = module.encode_domains(adjusted_batch)

            # Apply scaling and compute Euclidean norms
            scaled_latents = {}
            for domain_set, domain_data in latents.items():
                for domain_name in domain_set:
                    if domain_name == 'attr':
                        scaled_latents[domain_name] = domain_data[domain_name] * scaling_factor
                    else:  # Assuming 'v_latents'
                        scaled_latents[domain_name] = domain_data[domain_name] * (1 - scaling_factor)

            encoded_latents = {domain: module.gw_mod.gw_interfaces[domain].encode(scaled_latents[domain]) for domain in scaled_latents}

            for domain_name in encoded_latents.keys():
                    # Compute and accumulate the Euclidean norm for each domain
                    norm = torch.norm(encoded_latents[domain_name], dim=1).mean().item()
                    if domain_name == 'attr':
                        total_norm_attr += norm
                    else:  # Assuming 'v_latents'
                        total_norm_v_latents += norm

            total_batches += 1

    # Compute and store the average norm for each domain for the current scaling factor
    avg_norms_v_latents.append(total_norm_v_latents / total_batches)
    avg_norms_attr.append(total_norm_attr / total_batches)

# Convert lists to NumPy arrays for plotting
avg_norms_v_latents_np = np.array(avg_norms_v_latents)
avg_norms_attr_np = np.array(avg_norms_attr)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(scaling_factors.cpu().numpy(), avg_norms_v_latents_np, '-o', label='v_latents Norm')
plt.plot(scaling_factors.cpu().numpy(), avg_norms_attr_np, '-o', label='attr Norm')
plt.xlabel('Scaling Factor')
plt.ylabel('Average Euclidean Norm')
plt.title('Average Euclidean Norm over Scaling Factors')
plt.legend()
plt.grid(True)
plt.savefig("norms_vs_scaling_factors.png")
plt.show()
