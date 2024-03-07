from collections.abc import Mapping
import logging
from collections.abc import Callable
from typing import Any

import random

from datetime import datetime

import torch
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         RichProgressBar)
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import OmegaConf
from shimmer import load_structured_config
from shimmer.modules.global_workspace import (DeterministicGlobalWorkspace,
                                              SchedulerArgs, VariationalGlobalWorkspace)
from torch import set_float32_matmul_precision
import torch.nn.functional as F

from simple_shapes_dataset import DEBUG_MODE, PROJECT_DIR
from simple_shapes_dataset.config.root import Config
from simple_shapes_dataset.dataset import SimpleShapesDataModule
from simple_shapes_dataset.dataset.pre_process import (color_blind_visual_domain,
                                                       nullify_attribute_rotation)
from simple_shapes_dataset.logging import LogGWImagesCallback
from simple_shapes_dataset.modules.domains import load_pretrained_domains
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from collections import defaultdict



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

def sample_scaling_factor(a, temperature=1000):
    """
    Samples a scaling factor k from a uniform distribution and applies softmax with a specified temperature.
    Returns the scaling factors along with the method used ('binary' or 'softmax').
    """
    if torch.rand(1).item() < a:
        # Sample binary (0 or 1)
        k = torch.randint(0, 2, (1,)).float()
        softmax_values = torch.tensor([k, 1 - k])
        print(softmax_values," are the softmax values!!!!!\n\n\n\n")
        method = 'binary'
    else:
        # Sample uniformly from the entire range [0, 1]
        k = torch.rand(1)
        # Apply softmax with temperature
        softmax_values = F.softmax(torch.tensor([k, 1 - k]) * temperature, dim=0)
        method = 'softmax'
    return softmax_values[0].item(), softmax_values[1].item(), method



config = load_structured_config(
    "../config",
    Config,
    load_dirs=["local"],
    debug_mode=DEBUG_MODE,
)


seed_everything(config.seed, workers=True)

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
    domain_args=config.global_workspace.domain_args,
    additional_transforms=additional_transforms,
)

domain_modules = load_pretrained_domains(
    config.global_workspace.domains,
    config.global_workspace.encoders.hidden_dim,
    config.global_workspace.encoders.n_layers,
    config.global_workspace.decoders.hidden_dim,
    config.global_workspace.decoders.n_layers,
)

loss_coefs: dict[str, torch.Tensor] = {
    "demi_cycles": torch.tensor([1.0]),
    "cycles": torch.tensor([1.0]),
    "translations": torch.tensor([1.0]),
    "contrastives": torch.tensor([0.000001]),
}

module = DeterministicGlobalWorkspace(
    domain_modules,
    config.global_workspace.latent_dim,
    loss_coefs,
    config.training.optim.lr,
    config.training.optim.weight_decay,
    scheduler_args=SchedulerArgs(
        max_lr=config.training.optim.max_lr,
        total_steps=config.training.max_steps,
    ),
)

train_samples = data_module.get_samples("train", 32)
val_samples = data_module.get_samples("val", 32)
test_samples = data_module.get_samples("test", 32)
for domains in val_samples.keys():
    for domain in domains:
        val_samples[frozenset([domain])] = {
            domain: val_samples[domains][domain]
        }
        test_samples[frozenset([domain])] = {
            domain: test_samples[domains][domain]
        }
    break

callbacks: list[Callback] = [
    LearningRateMonitor(logging_interval="step"),
    LogGWImagesCallback(
        val_samples,
        log_key="images/val",
        mode="val",
        every_n_epochs=config.logging.log_val_medias_every_n_epochs,
    ),
    LogGWImagesCallback(
        val_samples,
        log_key="images/test",
        mode="test",
        every_n_epochs=None,
    ),
    LogGWImagesCallback(
        train_samples,
        log_key="images/train",
        mode="train",
        every_n_epochs=config.logging.log_train_medias_every_n_epochs,
    ),
]

if config.training.enable_progress_bar:
    callbacks.append(RichProgressBar())

wandb_logger = None
if config.wandb.enabled:
    gw_type = "var_gw" if config.global_workspace.is_variational else "gw"
    run_name = f"{gw_type}_z={config.global_workspace.latent_dim}"
    wandb_logger = WandbLogger(
        save_dir=config.wandb.save_dir,
        project=config.wandb.project,
        entity=config.wandb.entity,
        tags=["train_gw"],
        name="shapes_classifier_bigmodel_ais00",
    )
    wandb_logger.experiment.config.update(
        OmegaConf.to_container(config, resolve=True)
    )

    checkpoint_dir = (
        config.default_root_dir
        / f"{wandb_logger.name}-{wandb_logger.version}"
    )
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        )
    )


########################################################################   loading the pretrained fusion translators

        # Path to your checkpoint file
checkpoint_path = '/home/rbertin/simple-shapes-dataset-main/good_models/simple_shapes_fusion-wxu61lph/epoch=159.ckpt'

        # Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:0'))

        # Assuming module is your DeterministicGlobalWorkspace model
        # Load the state dict from the checkpoint into your model
module.load_state_dict(checkpoint['state_dict'])
module = module.to(torch.device("cuda:0"))


A=0.

########################################################################   classification head



class ClassificationHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Increasing the number of layers for more complexity
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        return self.fc5(x)

########################################################################   training loop



# Classification Head Initialization
classification_head = ClassificationHead(config.global_workspace.latent_dim, 3).to(torch.device("cuda:0"))  # Adjust encoded_output_dim as per your model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(classification_head.parameters()), lr=0.00002)  # Adjust learning rate as needed

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

def compute_scaled_latents(latents, a, temperature):
    k, one_minus_k, _ = sample_scaling_factor(a, temperature)
    scaled_latents = {}

    for domain_set, domain_data in latents.items():
        for domain_name in domain_set:
            domain_latents = domain_data[domain_name].clone()

            # Apply scaling factor based on the domain
            if domain_name == 'attr':
                scaling_factor = k  # or one_minus_k, depending on your requirement
            elif domain_name == 'v_latents':
                scaling_factor = one_minus_k  # or k, depending on your requirement
            else:
                raise ValueError(f"Unexpected domain name: {domain_name}")

            scaled_latent = domain_latents * scaling_factor
            print(f"Applying {scaling_factor} to {domain_name}")

            scaled_latents[domain_name] = scaled_latent

    return scaled_latents

def compute_loss_and_backpropagate(shape_info, global_workspace_vector, criterion, optimizer):

    shape_class_indices = torch.argmax(shape_info, dim=1)  # Assuming one-hot encoding

    # Compute loss
    shape_logits = classification_head(global_workspace_vector)
    loss = criterion(shape_logits, shape_class_indices)  # Use class indices
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

val_dataloader = data_module.val_dataloader()  # Load the validation data
# Evaluation Loop
device = torch.device("cuda:0")  # Or any other device you are using
evaluation_dataloader = tqdm(val_dataloader, desc="Evaluation")
total_correct_val = 0
total_samples_val = 0

# Define the fixed corruption vector
corruption_vector = torch.randn(config.global_workspace.latent_dim).to(torch.device("cuda:0"))  # Match the dimension

from torch.nn.functional import mse_loss

with torch.no_grad():
    total_correct = 0
    total_samples = 0
    
    for batch_tuple in evaluation_dataloader:
        # Extracting the batch from the tuple
        batch = batch_tuple[0]

        # Adjust the batch to match the training batch format
        adjusted_batch = {frozenset([key]): {key: value} for key, value in batch.items()}

        shape_info, adjusted_batch = prepare_data_for_training(adjusted_batch, device)
        if shape_info is None:
            print("Shape information not found in evaluation batch.")
            continue

        latents = module.encode_domains(adjusted_batch)


        # Apply corruption to a random latent domain
        random_domain = random.choice(list(latents.keys()))
        scale = torch.rand(1).to(device)
        latents[random_domain][next(iter(random_domain))] += scale * corruption_vector




        scaled_latents = compute_scaled_latents(latents, a=A, temperature=5)

        global_workspace_vector = module.encode(scaled_latents)

        # Decode the global workspace vector
        decoded_domains = module.decode(global_workspace_vector)

        adjusted_latents = {frozenset([key]): {key: value} for key, value in decoded_domains.items()}

        # Decode the domains to get the attribute vector
        decoded_attrs = module.decode_domains(adjusted_latents)

        # Select the first tensor from the list for 'attr' domain
        predicted_shape_info = decoded_attrs[frozenset(['attr'])]['attr'][0]

        # Compute accuracy
        correct_predictions = (predicted_shape_info.argmax(dim=1) == shape_info.argmax(dim=1)).sum().item()
        total_correct += correct_predictions
        total_samples += shape_info.size(0)

        # Exit after the first iteration
        break


    # Calculate and print the total evaluation accuracy
    accuracy = 100 * total_correct / total_samples
    print(f"Total Evaluation Accuracy: {accuracy:.2f}%")
