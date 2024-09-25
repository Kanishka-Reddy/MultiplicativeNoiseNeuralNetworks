import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from Diffusion_Models import SDM_DIFFUSION
import numpy as np
from collections import namedtuple
import wandb
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
import os

# Define the experiment parameters
Params = namedtuple('Params', [
    'dataset_name', 'input_size', 'output_size', 'nneurons', 'epochs', 'batch_size',
    'learning_rate', 'diffusion_noise', 'noise_type', 'use_wandb', 'device',
    'classification', 'transpose', 'use_projection_matrix', 'project_before_noise',
    'adjust_diffusion', 'use_bias_hidden', 'use_bias_output', 'act_func',
    'k_approach', 'norm_addresses', 'norm_values', 'norm_activations',
    'all_positive_weights', 'cifar10_mean', 'cifar10_std'
])


def run_experiment(width, noise_level):
    # Set up parameters
    params = Params(
        dataset_name='CIFAR10',
        input_size=3072,  # 32x32x3
        output_size=10,
        nneurons=[width],
        epochs=3000,
        batch_size=128,
        learning_rate=0.001,
        diffusion_noise=noise_level,
        noise_type='normal',
        use_wandb=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        classification=True,
        transpose=False,
        use_projection_matrix=False,
        project_before_noise=False,
        adjust_diffusion=False,
        use_bias_hidden=True,
        use_bias_output=True,
        act_func='relu',
        k_approach=None,
        norm_addresses=False,
        norm_values=False,
        norm_activations=False,
        all_positive_weights=False,
        cifar10_mean=(0.4914, 0.4822, 0.4465),
        cifar10_std=(0.2023, 0.1994, 0.2010)
    )

    # Create a unique name for the model
    model_name = f"CIFAR10_SDM_width{width}_noise{noise_level:.2f}"

    # Set up wandb
    wandb.init(project="cifar10-sdm-diffusion-extended", name=model_name, config=params._asdict())

    # Load CIFAR10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(params.cifar10_mean, params.cifar10_std)
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    # Initialize model
    model = SDM_DIFFUSION(params).to(params.device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # Training loop
    for epoch in range(params.epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        active_neurons = []
        pre_activations = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(params.device), target.to(params.device)
            data = data.view(data.size(0), -1)  # Flatten the images
            optimizer.zero_grad()
            output, activations = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Training-time metrics
            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            epoch_accuracy += pred.eq(target.view_as(pred)).sum().item()
            active_neurons.append((activations > 0).float().mean().item())
            pre_activations.append(model.net[model.neuron_layer_ind].layer.output.detach().cpu().numpy())

        # Log training metrics
        avg_loss = epoch_loss / len(train_loader)
        avg_accuracy = 100. * epoch_accuracy / len(train_loader.dataset)
        avg_active_neurons = np.mean(active_neurons)
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy,
            'avg_active_neurons': avg_active_neurons,
        })

        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(params.device), target.to(params.device)
                data = data.view(data.size(0), -1)  # Flatten the images
                output, _ = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_accuracy += pred.eq(target.view_as(pred)).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        avg_val_accuracy = 100. * val_accuracy / len(test_loader.dataset)
        wandb.log({
            'val_loss': avg_val_loss,
            'val_accuracy': avg_val_accuracy,
        })

        # Log pre-activation distribution (every 100 epochs)
        if epoch % 100 == 0:
            pre_act_flat = np.concatenate(pre_activations).flatten()
            fig, ax = plt.subplots()
            ax.hist(pre_act_flat, bins=100, density=True)
            ax.set_title(f"Pre-Activation Distribution (Epoch {epoch})")
            ax.set_xlabel("Pre-activation value")
            ax.set_ylabel("Density")
            wandb.log({f"pre_act_dist_epoch_{epoch}": wandb.Image(fig)})
            plt.close(fig)

    # Post-training analysis
    model.eval()
    post_train_metrics = {}

    # Covariance loss as a function of noise
    noise_levels = np.array([0.05, 0.1, 0.8, 1.5])
    cov_losses = []
    for noise in noise_levels:
        model.noise_layer.noise_amount = noise
        cov_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(params.device).view(data.size(0), -1)
                output, _ = model(data)
                cov_loss += ((output - data) ** 2).mean().item()
        cov_losses.append(cov_loss / len(test_loader))

    fig, ax = plt.subplots()
    ax.plot(noise_levels, cov_losses)
    ax.set_title("Covariance Loss vs Noise Level")
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Covariance Loss")
    post_train_metrics['cov_loss_vs_noise'] = wandb.Image(fig)
    plt.close(fig)

    # CIFAR10 reconstructions
    with torch.no_grad():
        sample_images, _ = next(iter(test_loader))
        sample_images = sample_images.to(params.device)
        reconstructed, _ = model(sample_images.view(sample_images.size(0), -1))
        reconstructed = reconstructed.view(sample_images.shape)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(5):
            axes[0, i].imshow(sample_images[i].cpu().permute(1, 2, 0))
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructed[i].cpu().permute(1, 2, 0))
            axes[1, i].axis('off')
        fig.suptitle("Original (top) vs Reconstructed (bottom)")
        post_train_metrics['reconstructions'] = wandb.Image(fig)
        plt.close(fig)

    # Receptive fields
    weights = model.net[model.neuron_layer_ind].layer.weight.detach().cpu()
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(16):
        rf = weights[i].view(3, 32, 32).permute(1, 2, 0)
        rf = (rf - rf.min()) / (rf.max() - rf.min())
        axes[i // 4, i % 4].imshow(rf)
        axes[i // 4, i % 4].axis('off')
    fig.suptitle("Sample Receptive Fields")
    post_train_metrics['receptive_fields'] = wandb.Image(fig)
    plt.close(fig)

    # Log all post-training metrics
    wandb.log(post_train_metrics)

    # Save the model
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(model.state_dict(), f'saved_models/{model_name}.pth')
    wandb.save(f'saved_models/{model_name}.pth')

    wandb.finish()


# Run experiments with different widths and noise levels
widths = [100]
noise_levels = [0.0, 0.05, 0.1, 0.8, 1.5, 3.0, 8.0]

for width in widths:
    for noise in noise_levels:
        print(f"\nRunning experiment with width {width} and noise level {noise}")
        run_experiment(width, noise)