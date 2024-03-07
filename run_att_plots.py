import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

def plot_train_vs_val(metrics, metrics_per_scale, output_dir, train_metric_key, val_metric_key, title, ylabel):
    epochs = list(range(1, len(metrics[train_metric_key]) + 1))
    plt.figure(facecolor='lightgrey', figsize=(10, 5))
    color = 'green' if 'v_latents' in train_metric_key else 'red'
    plt.plot(epochs, metrics[train_metric_key], label='Train', color=color, linestyle='-', marker='o')
    if -1 in metrics_per_scale:
        val_metrics = metrics_per_scale[-1][val_metric_key]
        plt.plot(epochs, val_metrics, label='Validation (-1 Scale)', color=color, linestyle='--', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()

def plot_across_scales(metrics_per_scale, output_dir, metric_key, title, ylabel):
    plt.figure(facecolor='lightgrey', figsize=(10, 5))
    scales = sorted(metrics_per_scale.keys())
    cmap = plt.get_cmap('Greens') if 'v_latents' in metric_key else plt.get_cmap('Reds')
    for i, scale in enumerate(scales):
        epochs = list(range(1, len(metrics_per_scale[scale][metric_key]) + 1))
        color = cmap(i / len(scales)) if scale != -1 else 'blue'
        # Format scale number to two decimals for the label
        scale_label = f"Scale {scale:.2f}" if scale != -1 else 'Scale -1'
        plt.plot(epochs, metrics_per_scale[scale][metric_key], label=scale_label, color=color)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    # Specify legend fontsize
    plt.legend(fontsize='small')  # You can adjust the value as needed, 'large' is roughly twice the default fontsize
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()

def plot_metric_comparison(metrics, output_dir, metric_keys, title, ylabel):
    plt.figure(facecolor='lightgrey', figsize=(10, 5))
    epochs = list(range(1, len(metrics[next(iter(metric_keys))]) + 1))
    for metric_key in metric_keys:
        color = 'green' if 'v_latents' in metric_key else 'red'
        linestyle = '-' if 'train' in metric_key else '--'
        plt.plot(epochs, metrics[metric_key], label=metric_key.replace("_", " ").capitalize(), color=color, linestyle=linestyle, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from attention model checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint)

    # Extract metrics and metrics_per_scale from the checkpoint
    metrics = checkpoint.get('metrics', {})
    metrics_per_scale = checkpoint.get('metrics_per_scale', {})

    # Generate and save plots for V_Latents and Attr across scales
    plot_across_scales(metrics_per_scale, args.output_dir, 'val_loss_v_latents', 'Validation Loss V_Latents Across Scales', 'Loss')
    plot_across_scales(metrics_per_scale, args.output_dir, 'val_accuracy_v_latents', 'Validation Accuracy V_Latents Across Scales', 'Accuracy')
    plot_across_scales(metrics_per_scale, args.output_dir, 'val_loss_attr', 'Validation Loss Attr Across Scales', 'Loss')
    plot_across_scales(metrics_per_scale, args.output_dir, 'val_accuracy_attr', 'Validation Accuracy Attr Across Scales', 'Accuracy')

    # Generate and save plots for Train vs Validation Loss and Accuracy for V_Latents and Attr
    plot_train_vs_val(metrics, metrics_per_scale, args.output_dir, 'train_loss_v_latents', 'val_loss_v_latents', 'Train vs Validation Loss V_Latents', 'Loss')
    plot_train_vs_val(metrics, metrics_per_scale, args.output_dir, 'train_accuracy_v_latents', 'val_accuracy_v_latents', 'Train vs Validation Accuracy V_Latents', 'Accuracy')
    plot_train_vs_val(metrics, metrics_per_scale, args.output_dir, 'train_loss_attr', 'val_loss_attr', 'Train vs Validation Loss Attr', 'Loss')
    plot_train_vs_val(metrics, metrics_per_scale, args.output_dir, 'train_accuracy_attr', 'val_accuracy_attr', 'Train vs Validation Accuracy Attr', 'Accuracy')

    # Generate and save plots for Validation Loss and Accuracy Comparison between V_Latents and Attr
    plot_metric_comparison(metrics_per_scale[-1], args.output_dir, ['val_loss_v_latents', 'val_loss_attr'], 'Validation Loss for Two Domains', 'Loss')
    plot_metric_comparison(metrics_per_scale[-1], args.output_dir, ['val_accuracy_v_latents', 'val_accuracy_attr'], 'Validation Accuracy for Two Domains', 'Accuracy')

    # Generate and save plots for Training Loss and Accuracy Comparison between V_Latents and Attr
    plot_metric_comparison(metrics, args.output_dir, ['train_loss_v_latents', 'train_loss_attr'], 'Training Loss for Two Domains', 'Loss')
    plot_metric_comparison(metrics, args.output_dir, ['train_accuracy_v_latents', 'train_accuracy_attr'], 'Training Accuracy for Two Domains', 'Accuracy')
