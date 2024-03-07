import logging
from collections.abc import Callable
from typing import Any

import glob

import torch
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
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
from collections.abc import Mapping
import logging
from collections.abc import Callable
from typing import Any

import random
import os

from datetime import datetime

import torch
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         RichProgressBar)
from torch import set_float32_matmul_precision
import torch.nn.functional as F

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



device = torch.device("cuda:0")

import sys

import os

# Get the fusion model name from an environment variable
fusion_model_name = os.getenv('FUSION_MODEL_NAME')
if fusion_model_name is None:
    raise ValueError("FUSION_MODEL_NAME environment variable not set")

# Get the attention model filename from an environment variable
model_filename = os.getenv('MODEL_FILENAME')
if model_filename is None:
    raise ValueError("MODEL_FILENAME environment variable not set")

# Construct the paths based on the environment variables
fusion_model_checkpoint_path = f'/home/rbertin/cleaned/simple-shapes-dataset/simple_shapes_fusion-{fusion_model_name}/epoch=*.ckpt'  # Adjust as needed to select the correct file
attention_model_checkpoint_path = f'models_fullrandom/{fusion_model_name}/{model_filename}'

print("fusion_model_checkpoint_path : ", fusion_model_checkpoint_path)
print("attention_model_checkpoint_path : ", attention_model_checkpoint_path)


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

# Load the fusion model checkpoint
# Here you might need logic similar to what we did in the bash script to select the latest checkpoint if there are multiple
latest_fusion_checkpoint = max(glob.glob(fusion_model_checkpoint_path), key=os.path.getctime)
fusion_checkpoint = torch.load(latest_fusion_checkpoint, map_location=torch.device('cuda:0'))
module.load_state_dict(fusion_checkpoint['state_dict'])
module = module.to(torch.device("cuda:0"))

A=0.5

##########################################################################   attention

class AttentionMechanism(nn.Module):
    def __init__(self, domain_dim, head_size):
        super().__init__()
        self.head_size = head_size
        self.query_layer = nn.Linear(domain_dim, head_size)
        self.key_layers = nn.ModuleDict({
            'v_latents': nn.Linear(domain_dim, head_size),
            'attr': nn.Linear(domain_dim, head_size)
        })
        self.gw_vector = torch.randn(domain_dim).to(device)  # Fixed random global workspace vector
        self.attention_scores = None  # Attribute to store the latest attention scores

    def forward(self, domain_encodings: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        # Initialize key_layers if not already done
        keys = {domain: self.key_layers[domain](encoding) for domain, encoding in domain_encodings.items()}

        query = self.query_layer(self.gw_vector)

        # Calculate dot products for each domain
        dot_products = [torch.sum(key_tensor * query, dim=1) for key_tensor in keys.values()]

        # Organize the dot products into a 2D tensor [number_of_domains, 2]
        dot_products_tensor = torch.stack(dot_products)

        self.attention_scores = torch.softmax(dot_products_tensor,dim=0)

        # Scale the input latents by attention scores and print before and after scaling for the first two pairs
        weighted_encodings = {}
        for idx, (domain, encoding) in enumerate(domain_encodings.items()):
            # Reshape attention scores to match the encoding shape for broadcasting
            scaled_latent = encoding * self.attention_scores[idx].unsqueeze(1).expand_as(encoding)
            weighted_encodings[domain] = scaled_latent

        return weighted_encodings

# Initialize the Attention Mechanism
domain_dim = 12  # Dimension of each domain-encoded vector
nb_weights = 2       # Dimension of the global workspace vector
attention_mechanism = AttentionMechanism(domain_dim, nb_weights).to(device)

# Load the attention model checkpoint
attention_checkpoint = torch.load(attention_model_checkpoint_path, map_location=torch.device('cuda:0'))
attention_mechanism.load_state_dict(attention_checkpoint['model_state_dict'])

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

def compute_scaled_latents(latents, attention_mechanism, device):
    # Ensure the attention mechanism is on the correct device
    attention_mechanism = attention_mechanism.to(device)

    # Move latents to the correct device and flatten the structure for the attention mechanism
    flat_latents = {domain: domain_data[domain].to(device) 
                for domain_set, domain_data in latents.items()
                for domain in domain_set}

    # Get weighted (or scaled) encodings using the attention mechanism
    weighted_encodings = attention_mechanism(flat_latents)

    # Re-structure the weighted encodings to match the original latents structure
    scaled_latents = {}
    for domain_set in latents:
        scaled_latents[domain_set] = {domain: weighted_encodings[domain] for domain in domain_set}

    return scaled_latents

with torch.no_grad():
    # Initialize storage for attention weights as empty dictionaries for each corruption scenario
    domain_attention_weights_v_latents_corrupted = {}
    domain_attention_weights_attr_corrupted = {}

    # Define corruption scales
    corruption_scales = torch.linspace(0, 1, steps=10).to(device)

    # Iterate over corruption scales
    for scale in tqdm(corruption_scales, desc='Corruption Scales'):
        print(f"\nProcessing scale: {scale.item()}")
        
        # Process only one batch of data for demonstration purposes
        batch_tuple = next(iter(data_module.val_dataloader()))
        batch = batch_tuple[0]

        adjusted_batch = {frozenset([key]): {key: value} for key, value in batch.items()}
        adjusted_batch = to_device(adjusted_batch, device)

        # Encode the domains
        latents = module.encode_domains(adjusted_batch)

        # Iterate over domains for separate corruption scenarios
        for corruption_target_domain in ['v_latents', 'attr']:
            print(f"\nCorrupting {corruption_target_domain} domain:")
            
            # Clone latents to avoid in-place modifications affecting the other scenario
            latents_corrupted = {k: {dk: dv.clone() for dk, dv in v.items()} for k, v in latents.items()}

            # Introduce corruption only to the target domain
            for domain in latents_corrupted.keys():
                domain_key = next(iter(domain))  # Extract the domain name from the frozenset
                if domain_key == corruption_target_domain:
                    batch_size = latents[domain][domain_key].size(0)  # Assuming latents[domain][domain_key] has shape [batch_size, latent_dim]
                    corruption_vectors = torch.randn(batch_size, config.global_workspace.latent_dim).to(torch.device("cuda:0"))
                    corruption_vectors = (corruption_vectors - corruption_vectors.mean(dim=1, keepdim=True)) / corruption_vectors.std(dim=1, keepdim=True)
                    corruption_vectors = corruption_vectors * 5.
                    print("corruption vectors : ",corruption_vectors.shape)
                    latents_corrupted[domain][domain_key] += scale * corruption_vectors
                    print(f"Corrupted {domain_key} sample (first element):", latents_corrupted[domain][domain_key][0][:5])

            # Apply attention mechanism and get scaled latents
            scaled_latents = compute_scaled_latents(latents_corrupted, attention_mechanism, device)

            # Record the attention scores
            attention_scores = attention_mechanism.attention_scores.detach().cpu().numpy()
            print(f"Attention scores (first elements): {attention_scores[:5]}")

            for idx, domain in enumerate(latents_corrupted.keys()):
                domain_name = next(iter(domain))
                # Initialize the nested dictionary structure if it's the first time encountering this domain
                target_dict = (domain_attention_weights_v_latents_corrupted if corruption_target_domain == 'v_latents'
                               else domain_attention_weights_attr_corrupted)

                if domain_name not in target_dict:
                    target_dict[domain_name] = {scale.item(): [] for scale in corruption_scales}

                # Append the attention score for the current domain and corruption scale
                target_dict[domain_name][scale.item()].append(attention_scores[idx])
            
            print(f"Collected attention scores for {corruption_target_domain} corruption scenario.")

    # Print a summary of the collected attention weights for each corruption scenario
    for scenario_name, attention_weights in [('v_latents_corrupted', domain_attention_weights_v_latents_corrupted),
                                             ('attr_corrupted', domain_attention_weights_attr_corrupted)]:
        print(f"\nSummary for {scenario_name}:")
        for domain, scales_scores in attention_weights.items():
            print(f"{domain}: Scales: {list(scales_scores.keys())}")
            for scale, scores in scales_scores.items():
                print(f"Scale {scale}: Mean score: {torch.tensor(scores).mean().item()}")

# Aggregate attention weights for each domain and scale for both corruption scenarios
for scenario_name, attention_weights in [('v_latents_corrupted', domain_attention_weights_v_latents_corrupted),
                                         ('attr_corrupted', domain_attention_weights_attr_corrupted)]:
    print(f"\nAggregating scores for {scenario_name}:")
    for domain_name, scales_scores in attention_weights.items():
        for scale_key, scores in scales_scores.items():
            # Convert scores list to tensor and calculate the mean
            scores_tensor = torch.tensor(scores)
            mean_score = scores_tensor.mean().item()  # Get the mean score
            # Update the scores with the mean value
            attention_weights[domain_name][scale_key] = mean_score
            print(f"{domain_name} at scale {scale_key}: Mean score: {mean_score}")

# Compute statistics and plot for each domain for both corruption scenarios
# Construct the directory name to save the plots, including both the fusion model name and the model filename
plot_dir = f"scoreplots_fullrandom/scoreplots_{fusion_model_name}_{model_filename}"

# Create the directory if it does not exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Extract the scales from the keys for attr and v_latents
attr_scales = list(map(float, domain_attention_weights_attr_corrupted['attr'].keys()))
v_latents_scales = list(map(float, domain_attention_weights_v_latents_corrupted['v_latents'].keys()))

# Reverse attr_scales and append zeros
scales_attr = attr_scales[::-1] + [0.0] * len(attr_scales)

# For v_latents_scales, append zeros without reversing
scales_v_latents = [0.0] * len(v_latents_scales) + v_latents_scales

# Extract scores, assuming you want to concatenate reversed attr scores and straight v_latents scores
attr_scores = list(domain_attention_weights_attr_corrupted['attr'].values())[::-1] + list(domain_attention_weights_v_latents_corrupted['attr'].values())
v_latents_scores = list(domain_attention_weights_attr_corrupted['v_latents'].values())[::-1] + list(domain_attention_weights_v_latents_corrupted['v_latents'].values())

# Ensure the scales array matches the length of the scores arrays
assert len(scales_attr) == len(attr_scores), "attr scales and scores arrays must have the same length"
assert len(scales_v_latents) == len(v_latents_scores), "v_latents scales and scores arrays must have the same length"


# Truncate scales to 2 decimals
scales_attr = np.around(scales_attr, 2)
scales_v_latents = np.around(scales_v_latents, 2)

# Creating the plot with aesthetic adjustments
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twiny()



# Add vertical line at the intersection of the two axes (where x tick is 0)
intersection_index = 10  # Assuming the intersection is at index 10
plt.axvline(x=intersection_index, color='k', linestyle='--')

# Center line index
center_index = 10
# Total number of spans on each side
total_spans = 10
# Incremental alpha increase per span
alpha_increment = 0.05

# Gradient effect to the left of the center line
for i in range(total_spans):
    ax1.axvspan(center_index - i - 1, center_index - i, color='lightcoral', alpha=(i+1)*alpha_increment)

# Gradient effect to the right of the center line
for i in range(total_spans):
    ax1.axvspan(center_index + i, center_index + i + 1, color='lightgreen', alpha=(i+1)*alpha_increment)


fig.canvas.draw()

ax1.plot(range(len(attr_scores)), attr_scores, 'r-', label='attr attention scores')
ax2.plot(range(len(v_latents_scores)), v_latents_scores, 'g-', label='v_latents attention scores')

# Set x-ticks and colors for attr corruption scale (red)
ax1.set_xticks(range(len(scales_attr)))
ax1.set_xticklabels(scales_attr, rotation=45, color='red')

# Set x-ticks and colors for v_latents corruption scale (green)
ax2.set_xticks(range(len(scales_v_latents)))
ax2.set_xticklabels(scales_v_latents, rotation=45, color='green')

# Set labels for the axes
ax1.set_xlabel('Attr Corruption Scale', color='red')
ax2.set_xlabel('V_Latents Corruption Scale', color='green')
ax1.set_ylabel('Mean Attention Score')

# Title and legend
plt.title('Mean Attention Scores vs. Corruption Scale')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Set the limits for both axes to be the same
ax1.set_xlim(0, len(attr_scores) - 1)
ax2.set_xlim(ax1.get_xlim())

plt.tight_layout()

# Your code to show the plot...
plt.savefig(f"{plot_dir}/plot.png")

