import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from params import CP_PATH

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def save_model_weights(model, filename, verbose=1, cp_folder=CP_PATH):
    if verbose:
        print(f'\n -> Saving weights to {os.path.join(cp_folder,filename)}\n')
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def load_model_weights(model, filename, verbose=1, cp_folder=CP_PATH, strict=True):
    if verbose:
        print(
            f'\n -> Loading weights from {os.path.join(cp_folder,filename)}\n')
    try:
        model.load_state_dict(os.path.join(cp_folder, filename), strict=strict)
    except BaseException:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    cp_folder,
                    filename),
                map_location='cpu'),
            strict=strict)
    return model


def plot_losses(infos):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.plot(infos['discriminator_loss'], label='discriminator loss', c='darkorange')
    ax1.set_ylabel('Discriminator Loss', color='darkorange', size=14)
    ax1.tick_params(axis='y', colors='darkorange')
    ax1.set_xlabel('Epochs', size=14)
    plt.grid(True)
    plt.legend(loc=(0.45, 0.95))

    ax2 = ax1.twinx()
    plt.plot(infos['generator_loss'], label='generator loss', c='dodgerblue')
    ax2.set_ylabel('Generator Loss', color='dodgerblue', size=14)
    ax2.tick_params(axis='y', colors='dodgerblue')
    plt.legend(loc=(0.45, 0.9))

    plt.title('Loss evolution', size=15)
    plt.show()