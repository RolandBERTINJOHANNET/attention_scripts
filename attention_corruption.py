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

        attention_scores = torch.softmax(dot_products_tensor,dim=0)

        # Scale the input latents by attention scores and print before and after scaling for the first two pairs
        weighted_encodings = {}
        for idx, (domain, encoding) in enumerate(domain_encodings.items()):
            # Reshape attention scores to match the encoding shape for broadcasting
            scaled_latent = encoding * attention_scores[idx].unsqueeze(1).expand_as(encoding)
            weighted_encodings[domain] = scaled_latent

        return weighted_encodings

# Initialize the Attention Mechanism
domain_dim = 12  # Dimension of each domain-encoded vector
nb_weights = 2       # head size
attention_mechanism = AttentionMechanism(domain_dim, nb_weights).to(device)

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

# Define the path to the checkpoint
checkpoint_path = 'classification_model.pth'
state_dict = torch.load(checkpoint_path, map_location=device)

# Initialize the ClassificationHead model
classification_head = ClassificationHead(input_dim=config.global_workspace.latent_dim, output_dim=3).to(device)
classification_head.load_state_dict(state_dict)
classification_head.eval()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(attention_mechanism.parameters()), lr=0.0005)  # Adjust learning rate as needed

num_epochs=10

training_losses = []
validation_losses = []

data_module.setup(stage='fit')  # This sets up the training dataset

def update_metrics(loss, logits, shape_info, metrics, loss_key, accuracy_key):
    # Update loss metrics
    metrics[loss_key].append(loss)  # Ensure loss is a single scalar value

    # Compute accuracy
    true_classes = torch.argmax(shape_info, dim=1)
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == true_classes).float().mean().item() * 100  # Convert to percentage

    # Update accuracy metrics
    metrics[accuracy_key].append(accuracy)


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

def compute_loss_and_backpropagate(shape_info_v_latents,shape_info_attr, global_workspace_vector_v_latents, global_workspace_vector_attr, classification_head, criterion, optimizer):
    shape_class_indices_v_latents = torch.argmax(shape_info_v_latents, dim=1)  # Assuming one-hot encoding
    shape_class_indices_attr = torch.argmax(shape_info_attr, dim=1)  # Assuming one-hot encoding

    # Compute loss for v_latents corrupted version
    shape_logits_v_latents = classification_head(global_workspace_vector_v_latents)
    loss_v_latents = criterion(shape_logits_v_latents, shape_class_indices_v_latents)

    # Compute loss for attr corrupted version
    shape_logits_attr = classification_head(global_workspace_vector_attr)
    loss_attr = criterion(shape_logits_attr, shape_class_indices_attr)

    # Aggregate losses and compute average
    total_loss = (loss_v_latents + loss_attr) / 2

    # Backpropagation on the average loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Return individual losses and the average loss for logging
    return loss_v_latents.item(), loss_attr.item(), total_loss.item()

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

def get_gl_vector(latents_corrupted, attention_mechanism, device, module):
    # Compute scaled latents
    scaled_latents = compute_scaled_latents(latents_corrupted, attention_mechanism, device)

    # Compute proper scaled latents for 'v_latents' and 'attr'
    proper_scaled_latents = {
        domain: scaled_latents[frozenset({'attr', 'v_latents'})][domain] 
        for domain in ['attr', 'v_latents']
    }

    # Encode to get the global workspace vector
    global_workspace_vector = module.encode(proper_scaled_latents)

    return global_workspace_vector

# Function added just to make the val loop a little less long.
def process_batch(scale, scale_key, latents, shape_info, current_epoch_metrics, device, attention_mechanism, module, classification_head, criterion):
    latents_v_latents_corrupted, shape_info_v_latents, latents_attr_corrupted,shape_info_attr = apply_corruption(latents, shape_info, scale, corruption_vector, device)
    global_workspace_vector_v_latents = get_gl_vector(latents_v_latents_corrupted, attention_mechanism, device, module)
    global_workspace_vector_attr = get_gl_vector(latents_attr_corrupted, attention_mechanism, device, module)

    shape_class_indices_attr = torch.argmax(shape_info_attr, dim=1)  # Assuming one-hot encoding
    shape_class_indices_v_latents = torch.argmax(shape_info_v_latents, dim=1)  # Assuming one-hot encoding

    shape_logits_v_latents = classification_head(global_workspace_vector_v_latents)
    loss_v_latents = criterion(shape_logits_v_latents, shape_class_indices_v_latents)

    shape_logits_attr = classification_head(global_workspace_vector_attr)
    loss_attr = criterion(shape_logits_attr, shape_class_indices_attr)

    # Update the metrics with the computed losses and accuracies
    update_metrics(loss_v_latents.item(), shape_logits_v_latents, shape_info_v_latents, current_epoch_metrics[scale_key], 'val_loss_v_latents', 'val_accuracy_v_latents')
    update_metrics(loss_attr.item(), shape_logits_attr, shape_info_attr, current_epoch_metrics[scale_key], 'val_loss_attr', 'val_accuracy_attr')



# Define the fixed corruption vector
corruption_vector = torch.randn(config.global_workspace.latent_dim).to(torch.device("cuda:0"))  # Match the dimension
corruption_vector = (corruption_vector - corruption_vector.mean()) / corruption_vector.std()
corruption_vector = corruption_vector * 5.
print("corruption vector : ",corruption_vector)


# Correct initialization on separate lines
accuracy_per_scale_per_epoch_v_latents, avg_loss_per_scale_per_epoch_v_latents = {}, {}
accuracy_per_scale_per_epoch_attr, avg_loss_per_scale_per_epoch_attr = {}, {}
avg_accuracy_per_epoch_v_latents, avg_accuracy_per_epoch_attr = [], []
avg_loss_per_epoch_v_latents, avg_loss_per_epoch_attr = [], []

preset_scales = [-1.] + np.linspace(0, 1, 15).tolist()

# Initialize storage for metrics outside the epoch loop
metrics_per_scale_per_epoch = {
    scale: {
        'val_loss_v_latents': [],
        'val_accuracy_v_latents': [],
        'val_loss_attr': [],
        'val_accuracy_attr': [],
    }for scale in preset_scales
}


# Initialization with lists
metrics = {
    'train_loss_v_latents': [],
    'train_loss_attr': [],
    'train_accuracy_v_latents': [],
    'train_accuracy_attr': [],
}

# Main Training Loop
for epoch in range(num_epochs):

    attention_mechanism.train()

    print(f"Epoch {epoch + 1}/{num_epochs}")

    device = torch.device("cuda:0")  # Or any other device you are using
    training_dataloader = tqdm(iter(data_module.train_dataloader()), desc=f"Training Epoch {epoch + 1}")

    # Temporary storage for current epoch metrics

# Temporary storage for current epoch metrics
    current_epoch_metrics = {
        'train_loss_v_latents': [],
        'train_loss_attr': [],
        'train_accuracy_v_latents': [],
        'train_accuracy_attr': [],
    }

    for batch in training_dataloader:
        batch = batch[0]
        batch = to_device(batch,torch.device("cuda:0"))
        batch = {k: v for k, v in batch.items() if k == frozenset({'v_latents', 'attr'})}
        latents = module.encode_domains(batch)

        # Introduce corruption
        scale = torch.rand(1).to(device)
        shape_info, batch = prepare_data_for_training(batch, device)
        if shape_info is None:
            print("Shape information not found in training batch.")
            continue

        latents_v_latents_corrupted, shape_info_v_latents, latents_attr_corrupted, shape_info_attr = apply_corruption(latents, shape_info, scale, corruption_vector, device)

        # Get the global workspace vectors
        global_workspace_vector_v_latents = get_gl_vector(latents_v_latents_corrupted, attention_mechanism, device, module)
        global_workspace_vector_attr = get_gl_vector(latents_attr_corrupted, attention_mechanism, device, module)

        # Compute losses and backpropagate
        loss_v_latents, loss_attr, avg_loss = compute_loss_and_backpropagate(shape_info_v_latents,shape_info_attr, global_workspace_vector_v_latents, global_workspace_vector_attr, classification_head, criterion, optimizer)

        # After encoding and obtaining logits
        logits_v_latents = classification_head(global_workspace_vector_v_latents)
        logits_attr = classification_head(global_workspace_vector_attr)

        # Update metrics for v_latents corrupted version
        update_metrics(loss_v_latents, logits_v_latents, shape_info_v_latents, current_epoch_metrics, 'train_loss_v_latents', 'train_accuracy_v_latents')

        # Update metrics for attr corrupted version
        update_metrics(loss_attr, logits_attr, shape_info_attr, current_epoch_metrics, 'train_loss_attr', 'train_accuracy_attr')

        training_dataloader.set_description(f"Epoch {epoch + 1} - Loss V_Latents: {current_epoch_metrics['train_loss_v_latents'][-1]:.4f}, Acc V_Latents: {current_epoch_metrics['train_accuracy_v_latents'][-1]:.2f}%, Loss Attr: {current_epoch_metrics['train_loss_attr'][-1]:.4f}, Acc Attr: {current_epoch_metrics['train_accuracy_attr'][-1]:.2f}%")


    # Compute averages for the current epoch and append to metrics
    for key in current_epoch_metrics:
        metrics[key].append(np.mean(current_epoch_metrics[key]))


    #VALIDATION LOOP
    # Temporary storage for current epoch metrics
    current_epoch_metrics = {
        scale: {
            'val_loss_v_latents': [],
            'val_accuracy_v_latents': [],
            'val_loss_attr': [],
            'val_accuracy_attr': [],
        } for scale in preset_scales
    }
    attention_mechanism.eval()

    with torch.no_grad():
        for scale in preset_scales:  # Add -1 to your preset scales to handle it specifically
            for batch_tuple in tqdm(iter(data_module.val_dataloader()), desc=f"Validation Epoch {epoch + 1}"):
                batch = batch_tuple[0]

                # Prepare the batch as before
                adjusted_batch = {}
                for key, value in batch.items():
                    adjusted_batch[frozenset([key])] = {key: value}
                shape_info, adjusted_batch = prepare_data_for_training(adjusted_batch, device)
                if shape_info is None:
                    continue
                latents = module.encode_domains(adjusted_batch)

                # Combine 'v_latents' and 'attr' under a single key
                latents = {
                    frozenset({'v_latents', 'attr'}): {
                        'v_latents': latents[frozenset({'v_latents'})]['v_latents'],
                        'attr': latents[frozenset({'attr'})]['attr']
                    }
                }

                if scale == -1.0:
                    # When scale is -1, sample multiple scales for this batch
                    for _ in range(10):  # Sample 10 different scales
                        sampled_scale = torch.rand(1).item()  # Sample a new scale
                        process_batch(sampled_scale, scale, latents, shape_info, current_epoch_metrics, device, attention_mechanism, module, classification_head, criterion)
                else:
                    # Process the batch with the given scale
                    process_batch(scale, scale, latents, shape_info, current_epoch_metrics, device, attention_mechanism, module, classification_head, criterion)


            avg_loss_v_latents = np.mean(current_epoch_metrics[scale]['val_loss_v_latents'])
            avg_accuracy_v_latents = np.mean(current_epoch_metrics[scale]['val_accuracy_v_latents'])
            avg_loss_attr = np.mean(current_epoch_metrics[scale]['val_loss_attr'])
            avg_accuracy_attr = np.mean(current_epoch_metrics[scale]['val_accuracy_attr'])

            # Store these averages in metrics_per_scale_per_epoch for the current epoch and scale
            metrics_per_scale_per_epoch[scale]['val_loss_v_latents'].append(avg_loss_v_latents)
            metrics_per_scale_per_epoch[scale]['val_accuracy_v_latents'].append(avg_accuracy_v_latents)
            metrics_per_scale_per_epoch[scale]['val_loss_attr'].append(avg_loss_attr)
            metrics_per_scale_per_epoch[scale]['val_accuracy_attr'].append(avg_accuracy_attr)




# Update the checkpoint dictionary to include metrics and metrics_per_scale
checkpoint = {
    'model_state_dict': attention_mechanism.state_dict(),
    'corruption_vector': corruption_vector,
    'metrics': metrics, # Add the metrics dictionary
    'metrics_per_scale': metrics_per_scale_per_epoch  # Add the metrics_per_scale dictionary
}


# Save the updated checkpoint
torch.save(checkpoint, 'attention_run.pt')
