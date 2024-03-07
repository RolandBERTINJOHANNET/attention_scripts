import logging
from collections.abc import Callable
from typing import Any

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


config = load_config(
    "../config",
    load_files=["train_gw.yaml"],
    debug_mode=DEBUG_MODE,
)

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


import random

# Generate a random integer between 0 and 99999 (you can choose your range)
random_seed = random.randint(0, 99999)
print("seed : ",random_seed)


# Use the generated random seed
seed_everything(random_seed, workers=True)

########################################################################   loading the pretrained fusion translators

        # Path to your checkpoint file (this is meant to be run from the train_clf_and_att.sh script
checkpoint_path = os.getenv('CHECKPOINT_PATH')

        # Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:0'))

        # Assuming module is your DeterministicGlobalWorkspace model
        # Load the state dict from the checkpoint into your model
module.load_state_dict(checkpoint['state_dict'])
module = module.to(torch.device("cuda:0"))

print("just loaded the model ;)")

A=.5

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
optimizer = torch.optim.Adam(list(classification_head.parameters()), lr=0.0002)  # Adjust learning rate as needed

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

def compute_loss_and_backpropagate(shape_info, global_workspace_vector, criterion, optimizer):

    shape_class_indices = torch.argmax(shape_info, dim=1)  # Assuming one-hot encoding

    # Compute loss
    shape_logits = classification_head(global_workspace_vector)
    loss = criterion(shape_logits, shape_class_indices)  # Use class indices
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Main Training Loop
for epoch in range(num_epochs):
    total_train_loss = 0
    total_correct = 0
    total_samples = 0
    print(f"Epoch {epoch + 1}/{num_epochs}")
    total_train_loss = 0
    device = torch.device("cuda:0")  # Or any other device you are using
    training_dataloader = tqdm(iter(data_module.train_dataloader()), desc=f"Training Epoch {epoch + 1}")
    for batch in training_dataloader:
        batch = batch[0]
        batch = to_device(batch,torch.device("cuda:0"))
        batch = {k: v for k, v in batch.items() if k == frozenset({'v_latents', 'attr'})}
        latents = module.encode_domains(batch)
        shape_info, batch = prepare_data_for_training(batch, device)
        if shape_info is None:
            print("Shape information not found in training batch.")
            continue
        scaled_latents = compute_scaled_latents(latents, a=A, temperature=5)
        global_workspace_vector = module.encode(scaled_latents)
        loss = compute_loss_and_backpropagate(shape_info, global_workspace_vector, criterion, optimizer)
        total_train_loss += loss

        # Calculate accuracy
        shape_logits = classification_head(global_workspace_vector)
        predicted = torch.argmax(shape_logits, dim=1)
        correct = (predicted == shape_info.argmax(dim=1)).sum().item()
        total_correct += correct
        total_samples += shape_info.size(0)

        # Update tqdm description with current loss and accuracy
        accuracy = 100 * total_correct / total_samples
        training_dataloader.set_description(f"Training Epoch {epoch + 1} - Loss: {loss:.4f}, Acc: {accuracy:.2f}%")

    avg_train_loss = total_train_loss / len(iter(data_module.train_dataloader()))
    training_losses.append(avg_train_loss)

    # Calculate training accuracy
    training_accuracy = 100 * total_correct / total_samples

    print(f"Average Training Loss: {avg_train_loss}")


    # Validation Loop
    classification_head.eval()  # Set classification head to evaluation mode
    total_val_loss = 0
    total_correct_val = 0
    total_samples_val = 0

    with torch.no_grad():
        for batch_tuple in tqdm(iter(data_module.val_dataloader()), desc=f"Validation Epoch {epoch + 1}"):
            # Extracting the batch from the tuple
            batch = batch_tuple[0]

            # Adjust the batch to match the training batch format
            adjusted_batch = {}
            for key, value in batch.items():
                adjusted_batch[frozenset([key])] = {key: value}

            # Proceed as in the training loop
            shape_info, adjusted_batch = prepare_data_for_training(adjusted_batch, device)
            if shape_info is None:
                print("Shape information not found in validation batch.")
                continue

            latents = module.encode_domains(adjusted_batch)
            scaled_latents = compute_scaled_latents(latents, a=A, temperature=5)
            global_workspace_vector = module.encode(scaled_latents)

            # Compute loss with classification head
            shape_logits = classification_head(global_workspace_vector)
            # Convert shape_info to class indices
            shape_info = torch.argmax(shape_info, dim=1)
            loss = criterion(shape_logits, shape_info)
            total_val_loss += loss.item()

            predicted_val = torch.argmax(shape_logits, dim=1)
            correct_val = (predicted_val == shape_info).sum().item()
            total_correct_val += correct_val
            total_samples_val += shape_info.size(0)

    avg_val_loss = total_val_loss / len(iter(data_module.val_dataloader()))
    validation_losses.append(avg_val_loss)

    validation_accuracy = 100 * total_correct_val / total_samples_val

    print(f"Average Validation Loss: {avg_val_loss}")

# Save the trained classification model
torch.save(classification_head.state_dict(), 'classification_model.pth')
