import torch
import matplotlib.pyplot as plt
import numpy as np

from params import NUM_CLASSES, CLASSES


def generate(generator, noise=None, classes=None, n=5, n_plot=0, latent_dim=128, nb_classes=120, show_title=False):
    if noise is None:
        noise = torch.cuda.FloatTensor(n, latent_dim, 1, 1).normal_(0, 1)
    if classes is None:
        classes = torch.from_numpy(np.random.randint(0, nb_classes, size=n)).long().cuda()
        
    generated_images = generator(noise, classes).add(1).div(2)
    images = generated_images.cpu().clone().detach().numpy().transpose(0, 2, 3, 1)
    
    if n_plot:
        plt.figure(figsize=(15, 3 * n_plot//5))
        for i in range(n_plot):
            plt.subplot(n_plot//5, 5, i+1)
            plt.imshow(images[i])
            plt.axis('off')
            if show_title:
                plt.title(CLASSES[int(classes[i])])
        plt.show()
    
    return generated_images


def latent_walk(generator, n=10, latent_dim=128, a=None, b=None):
    
    if a is None:
        a = np.random.normal(size=latent_dim)
    if b is None:
        b = np.random.normal(size=latent_dim)
        
    classes = torch.from_numpy(np.random.randint(0, NUM_CLASSES, size=1)).long().cuda()
    plt.figure(figsize=(n * 3, 3))
    
    for j in range(n+1):
        noise = j / n * a + (1 - j / n) * b
        noise = noise / np.linalg.norm(noise)  # redresse ?        
        noise = torch.from_numpy(noise).view((1, latent_dim, 1, 1)).float().cuda()
        
        plt.subplot(1, n+1, j+1)
        img = generate(generator, noise=noise, classes=classes, latent_dim=latent_dim)
        img = img.cpu().clone().detach().numpy().transpose(0, 2, 3, 1).squeeze()

        plt.axis('off')
        plt.imshow(img)

    plt.show()