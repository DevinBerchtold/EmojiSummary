# import os
# import json
# import re # Import the regular expression module
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from emoji_ai import latest_checkpoint, load_losses

WINDOW = 20
CLIP_FRACTION = 0.05

NAME = 'emogist'
MODEL = 'llama3.2-1b'
VERSION = 3

BASE_PATH = 'C:/AI/Emogist/'
DIRECTORY = f'{BASE_PATH}{NAME}_{MODEL}_v{VERSION}'
OUTPUT_DIRECTORY = f'{BASE_PATH}{NAME}_{MODEL}_v{VERSION}/checkpoints'

def main():
    checkpoint_dir, checkpoint_num = latest_checkpoint(OUTPUT_DIRECTORY)

    train_epochs, train_losses, _, eval_epochs, eval_losses, _ = load_losses(checkpoint_dir)

    plt.figure(figsize=(12, 7)) # Adjusted figure size
    averages = []

    # Plot regular loss if data exists
    if train_epochs and train_losses:
        plt.scatter(train_epochs, train_losses, alpha=0.3, label='Raw Loss values', s=10)
        window_size = min(WINDOW, len(train_losses))
        for i in range(1, len(train_losses) + 1): # Moving average
            window = train_losses[max(0, i - window_size) : i]
            averages.append(sum(window) / len(window))
        
        plt.plot(train_epochs, averages, label=f'Train Loss MA (window={window_size})')
        plt.axhline(y=min(averages), linestyle='--', label=f'Min Train Loss: {min(averages):.4f}')

    # Plot evaluation loss if data exists
    if eval_epochs and eval_losses:
        plt.plot(eval_epochs, eval_losses, label='Eval Loss')
        plt.axhline(y=min(eval_losses), linestyle='--', label=f'Min Eval Loss: {min(eval_losses):.4f}')
    
    plt.xlim(0, max(max(train_epochs, eval_epochs)))
    # Max is value at CLIP_FRACTION of the total time
    plt.ylim(min(train_losses), averages[int(len(averages) * CLIP_FRACTION)])

    # Add labels and styling
    plt.title(f'Training & Evaluation Loss - Checkpoint {checkpoint_num}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.4, linestyle=':')
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()