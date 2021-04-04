import torch
# import matplotlib
# matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def plot_images_labels(x: torch.tensor, label, export_img, title: str = '', nrow=8, padding=2, normalize=False, pad_value=0):
    """Plot separate images of shape (H x W) colored by their binary label."""

    grid = make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.cpu().numpy()

    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    # plt.imshow(x[i].squeeze(), cmap='gray', interpolation='nearest')

    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Border color:
    if label == 0:
        color = 'green'
    elif label == 1:
        color = 'red'
    else:
        print('Invalid label assigned!')
    plt.setp(ax.spines.values(), color=color, linewidth=5)
    # plt.setp([ax.get_xticklines(), ax.get_yticklines()], color='green')

    if not (title == ''):
        plt.title(title)

    plt.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
