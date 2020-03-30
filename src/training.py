import torch
import matplotlib.pyplot as plt
import numpy as np

from params import NUM_CLASSES


def generate(generator, noise=None, races=None, n=5, n_plot=0, latent_dim=128):
    if noise is None:
        noise = torch.cuda.FloatTensor(n, latent_dim, 1, 1).normal_(0, 1)
    if races is None:
        races = torch.from_numpy(np.random.randint(0, NUM_CLASSES, size=n)).long().cuda()
        
    generated_images = generator(noise, races).add(1).div(2)
    images = generated_images.cpu().clone().detach().numpy().transpose(0, 2, 3, 1)
    
    if n_plot:
        plt.figure(figsize=(15, 3 * n_plot//5))
        for i in range(n_plot):
            plt.subplot(n_plot//5, 5, i+1)
            plt.imshow(images[i])
            plt.axis('off')
        plt.show()
    
    return generated_images


def latent_walk(generator, n=10, latent_dim=128):
    
    a = np.random.normal(size=latent_dim)
    b = np.random.normal(size=latent_dim)
    race = torch.from_numpy(np.random.randint(0, NUM_CLASSES, size=1)).long().cuda()
    
    plt.figure(figsize=(n * 3, 3))
    
    for j in range(n+1):
        noise = j / n * a + (1 - j / n) * b
        noise = noise / np.linalg.norm(noise)
        noise = torch.from_numpy(noise).view((1, latent_dim, 1, 1)).float().cuda()
        
        plt.subplot(1, n+1, j+1)
        img = generate(generator, noise=noise, races=race, latent_dim=latent_dim)
        img = img.cpu().clone().detach().numpy().transpose(0, 2, 3, 1).squeeze()

        plt.axis('off')
        plt.imshow(img)

    plt.show()