import json
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

# Define the 10 official ARC colors
ARC_COLORMAP = colors.ListedColormap([
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
])

def plot_grid(ax, grid, title=""):
    """Plots a single ARC grid with the official colormap."""
    norm = colors.Normalize(vmin=0, vmax=9)
    ax.imshow(np.array(grid), cmap=ARC_COLORMAP, norm=norm)
    ax.grid(True, which='both', color='white', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)

def plot_task(task):
    """Plots all training and test pairs for a given ARC task."""
    num_train = len(task['train'])
    num_test = len(task['test'])
    num_total = num_train + num_test
    fig, axs = plt.subplots(2, num_total, figsize=(3 * num_total, 6))
    
    for i, pair in enumerate(task['train']):
        plot_grid(axs[0, i], pair['input'], f"Train {i} Input")
        plot_grid(axs[1, i], pair['output'], f"Train {i} Output")

    for i, pair in enumerate(task['test']):
        plot_grid(axs[0, num_train + i], pair['input'], f"Test {i} Input")
        if 'output' in pair:
            plot_grid(axs[1, num_train + i], pair['output'], f"Test {i} Output")
        else:
            axs[1, num_train + i].axis('off')
            axs[1, num_train + i].set_title(f"Test {i} Output (Predict)")
    
    plt.tight_layout()
    plt.show()



def dsl_rotate_90(grid):
    return np.rot90(grid, 1)

def dsl_flip_horizontal(grid):
    return np.fliplr(grid)

def dsl_flip_vertical(grid):
    return np.flipud(grid)

# Our DSL is a dictionary mapping function names to functions
DSL = {
    'rotate_90': dsl_rotate_90,
    'flip_h': dsl_flip_horizontal,
    'flip_v': dsl_flip_vertical,
}

def load_task(task_path):
    with open(task_path, 'r') as f:
        return json.load(f)

# Example usage
task_file = '/Users/hpathak/dev/ARC-AGI/data/training/007bbfb7.json'
task = load_task(task_file)
plot_task(task)
get_first_example = task['train'][0]

print(get_first_example)


