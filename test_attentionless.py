import logging
from collections.abc import Callable
from typing import Any

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

def sample_scaling_factors(a, batch_size, temperature=1000):
    # Binary mask: 1 for binary scaling factor, 0 for uniform scaling factor
    binary_mask = torch.rand(batch_size) < a

    # Binary scaling factors
    binary_factors = torch.randint(0, 2, (batch_size,)).float()
    binary_softmax = torch.stack([binary_factors, 1 - binary_factors], dim=1)

    # Uniform samples for the entire range [0, 1]
    uniform_samples = torch.rand(batch_size)

    # Prepare uniform samples for softmax
    uniform_for_softmax = torch.stack([uniform_samples, 1 - uniform_samples], dim=1)

    # Softmax scaling factors for uniform samples
    uniform_softmax = F.softmax(uniform_for_softmax * temperature, dim=1)

    # Choose scaling factors based on the binary mask
    scaling_factors = torch.where(binary_mask.unsqueeze(-1), binary_softmax, uniform_softmax).to(torch.device("cuda:0"))

    # Get indices of binary and softmax samples
    binary_indices = torch.where(binary_mask)[0]
    softmax_indices = torch.where(~binary_mask)[0]

    binary_scaling_factors = scaling_factors[binary_indices]
    softmax_scaling_factors = scaling_factors[softmax_indices]


    return {
        'binary': (binary_scaling_factors[:, 0], binary_scaling_factors[:, 1], binary_indices),
        'softmax': (softmax_scaling_factors[:, 0], softmax_scaling_factors[:, 1], softmax_indices)
    }


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

device=torch.device("cuda:0")

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

########################################################################   inference loop
criterion = nn.CrossEntropyLoss()


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
    # Extracting batch size assuming all domains have the same batch size
    batch_size = next(iter(next(iter(latents.values())).values())).shape[0]

    scaling_factors = sample_scaling_factors(a, batch_size, temperature)
    scaled_latents = {}

    for domain_set, domain_data in latents.items():
        for domain_name in domain_set:
            domain_latents = domain_data[domain_name].clone()

            for scale_type, (k, one_minus_k, indices) in scaling_factors.items():
                # Apply scaling and select the subset of latents
                scaling_factor = k if domain_name == 'attr' else one_minus_k
                scaled_latents_subset = domain_latents[indices] * scaling_factor.unsqueeze(-1)
                scaled_latents_subset = scaled_latents_subset.to(domain_latents.dtype)

                # Initialize or update the scaled_latents for this domain_name
                if domain_name not in scaled_latents:
                    scaled_latents[domain_name] = torch.zeros_like(domain_latents)

                # Update the relevant indices of the scaled_latents
                scaled_latents[domain_name][indices] = scaled_latents_subset

    return scaled_latents

def apply_corruption(latents, shape_info, scale, corruption_vector, device):
    # Initialize structures for corrupted latents
    latents_v_latents_corrupted = {}
    latents_attr_corrupted = {}

    # Extract the combined key and the tensors for each domain
    combined_key = next(iter(latents.keys()))
    v_latents_tensor = latents[combined_key]['v_latents']
    attr_tensor = latents[combined_key]['attr']

    # Determine the split index
    split_index = v_latents_tensor.size(0) // 2

    # Apply corruption to the first half of v_latents
    v_latents_corrupted = v_latents_tensor[:split_index] + scale * corruption_vector.to(device)

    # Apply corruption to the second half of attr
    attr_corrupted = attr_tensor[split_index:] + scale * corruption_vector.to(device)

    # Split the shape_info tensor accordingly
    shape_info_v_latents = shape_info[:split_index]
    shape_info_attr = shape_info[split_index:]

    # Populate the structures with corrupted and original versions
    latents_v_latents_corrupted[combined_key] = {'v_latents': v_latents_corrupted, 'attr': attr_tensor[:split_index]}
    latents_attr_corrupted[combined_key] = {'v_latents': v_latents_tensor[split_index:], 'attr': attr_corrupted}

    return latents_v_latents_corrupted, shape_info_v_latents, latents_attr_corrupted, shape_info_attr



def get_gl_vector(latents_corrupted, device, module):
    
    # Compute proper scaled latents for 'v_latents' and 'attr'
    latents_corrupted = {
        domain: latents_corrupted[frozenset({'attr', 'v_latents'})][domain] 
        for domain in ['attr', 'v_latents']
    }
    # Encode to get the global workspace vector
    global_workspace_vector = module.encode(latents_corrupted)
    
    return global_workspace_vector


import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser(description='Process fusion models.')
parser.add_argument('--fusion_model', type=str, help='Name of the fusion model')
parser.add_argument('--output_csv_path', type=str, help='Path to save the output CSV file')
args = parser.parse_args()

# Use the fusion model name to construct the model directory path
model_dir = os.path.join('models', args.fusion_model)

# Now, you can list all .pth files in the specified model directory
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') and "classification" in f]

output_csv_path = args.output_csv_path


fieldnames = [
    'model',  # To match the 'model' key in your results dictionary
    'avg_val_loss'  # For the average validation loss
]

# Adding field names for scales in order
preset_scales = [0, 0.25, 0.5, 0.75, 1]
for scale in preset_scales:
    fieldnames.append(f'val_accuracy_attr_scale_{scale}')
    fieldnames.append(f'val_accuracy_v_latents_scale_{scale}')

# Ensure to use the CSV writer with these fieldnames
import csv

# Open the CSV file for writing
with open(output_csv_path, mode='w', newline='') as file:
    # Assuming 'results_file_path' is your CSV file path
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    preset_scales = [0, 0.25, 0.5, 0.75, 1]

    # Loop over each model file
    for model_file in model_files:
        # Load model as before, set to eval mode, etc.
        checkpoint_path = os.path.join(model_dir, model_file)
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Initialize the model and load the state dict
        classification_head = ClassificationHead(input_dim=config.global_workspace.latent_dim, output_dim=3).to(device)
        classification_head.load_state_dict(state_dict)
        classification_head.eval()

        # Prepare for storing results for each scale
        results = {'model': model_file, 'avg_val_loss': None}
        for scale in preset_scales:
            results[f'val_accuracy_attr_scale_{scale}'] = 0
            results[f'val_accuracy_v_latents_scale_{scale}'] = 0

        total_val_loss = 0
        total_samples_val = 0


        with torch.no_grad():
            for scale in preset_scales:
                total_correct_val_attr = 0
                total_correct_val_v_latents = 0
                total_samples_val_scale = 0
                for _ in tqdm(list(range(1)), desc=f"Validation for Scale {scale}"):
                    #temporarily define outside the loop jus tto see
                    # Define the fixed corruption vector
                    corruption_vector = torch.randn(config.global_workspace.latent_dim).to(torch.device("cuda:0"))  # Match the dimension
                    corruption_vector = (corruption_vector - corruption_vector.mean()) / corruption_vector.std()
                    corruption_vector = corruption_vector * 5.

                    for batch_tuple in iter(data_module.val_dataloader()):
                        batch = batch_tuple[0]

                        # Prepare the batch as before
                        adjusted_batch = {}
                        for key, value in batch.items():
                            adjusted_batch[frozenset([key])] = {key: value}
                        shape_info, adjusted_batch = prepare_data_for_training(adjusted_batch, device)
                        if shape_info is None:
                            continue

                        latents = module.encode_domains(adjusted_batch)
                        scaled_latents = compute_scaled_latents(latents, a=A, temperature=5)

                        # Combine 'v_latents' and 'attr' under a single key
                        latents = {
                            frozenset({'v_latents', 'attr'}): {
                                'v_latents': latents[frozenset({'v_latents'})]['v_latents'],
                                'attr': latents[frozenset({'attr'})]['attr']
                            }
                        }
                        latents_v_latents_corrupted, shape_info_v_latents, latents_attr_corrupted, shape_info_attr = apply_corruption(latents, shape_info, scale, corruption_vector, device)

                        # Process v_latents_corrupted version
                        global_workspace_vector_v_latents = get_gl_vector(latents_v_latents_corrupted, device, module)
                        shape_class_indices_v_latents = torch.argmax(shape_info_v_latents, dim=1)
                        logits_v_latents = classification_head(global_workspace_vector_v_latents)
                        loss_v_latents = criterion(logits_v_latents, shape_class_indices_v_latents)
                        total_val_loss += loss_v_latents.item()
                        correct_val_v_latents = (torch.argmax(logits_v_latents, dim=1) == shape_class_indices_v_latents).sum().item()
                        total_correct_val_v_latents += correct_val_v_latents

                        # Process attr_corrupted version
                        global_workspace_vector_attr = get_gl_vector(latents_attr_corrupted, device, module)
                        shape_class_indices_attr = torch.argmax(shape_info_attr, dim=1)
                        logits_attr = classification_head(global_workspace_vector_attr)
                        loss_attr = criterion(logits_attr, shape_class_indices_attr)
                        total_val_loss += loss_attr.item()
                        correct_val_attr = (torch.argmax(logits_attr, dim=1) == shape_class_indices_attr).sum().item()
                        total_correct_val_attr += correct_val_attr

                        # Since the batches are now halved, adjust the total sample count accordingly
                        total_samples_val_scale += shape_info_v_latents.size(0)

                # Calculate accuracy for this scale and update results dictionary
                results[f'val_accuracy_attr_scale_{scale}'] = 100 * total_correct_val_attr / total_samples_val_scale
                results[f'val_accuracy_v_latents_scale_{scale}'] = 100 * total_correct_val_v_latents / total_samples_val_scale
                total_samples_val += total_samples_val_scale

        avg_val_loss = total_val_loss / len(iter(data_module.val_dataloader())) / len(preset_scales)
        results['avg_val_loss'] = avg_val_loss

        print("results : ",results)


        # Log to CSV with updated results structure
        writer.writerow(results)
        print(f"Results for {model_file} written to CSV.")
