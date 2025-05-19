# import os
# import json
# import re # Import the regular expression module
import matplotlib.pyplot as plt
import numpy as np # Import numpy for percentile calculation

from emoji_ai import latest_checkpoint, load_losses, parent_name

plt.style.use('dark_background')

WINDOW = 20
CLIP_PERCENTILE = 98 # Use 98th percentile for upper y-limit instead of fraction
PLOT_TRAINING = False
PLOT_EVALUATION = True

# --- List of Directories ---
# Define the base path and specific experiment variations
BASE_PATH = 'C:/AI/Emogist/'
EXPERIMENTS = [ # Name, model, version  f'{BASE_PATH}{name}_{model}_v{version}/checkpoints'
    # ('emogist', 'llama3.2-1b', 0),
    ('emogist', 'llama3.2-1b', 1),
    # ('emogist', 'llama3.2-1b', 2),
    ('emogist', 'llama3.2-1b', 3),
    # ('emogist', 'llama3.2-1b', 4),
    ('emogist', 'llama3.2-1b', 5),
    ('emogist', 'llama3.2-1b', 6),
    ('emogist', 'llama3.2-3b', 7),
    ('emogist', 'llama3.2-3b', 8),
    ('emogist', 'llama3.2-3b', 9),
    ('emogist', 'llama3.2-3b', 10),
]

def main():
    plt.figure(figsize=(15, 8)) # Adjusted figure size for potentially more lines

    all_min_loss = float('inf')
    all_max_epoch = 0
    all_losses_for_ylim = [] # Collect relevant losses to determine y-axis limit

    # Loop through each directory provided
    for n, (name, model, version) in enumerate(EXPERIMENTS):
        directory = f"{BASE_PATH}{name}_{model}_v{version}/checkpoints"
        print("-" * 20)
        print(f"Processing directory: {directory}")

        checkpoint_dir, checkpoint_num = latest_checkpoint(directory)

        train_epochs, train_losses, train_steps, eval_epochs, eval_losses, eval_steps = load_losses(checkpoint_dir)

        # Get a label for the legend
        label_prefix = parent_name(directory).split('_')[-1] # Use helper function for label
        print(f"  Plotting data for: {label_prefix} (Step {checkpoint_num})")
        color = f'C{n}'

        # Plot regular loss if data exists
        if train_epochs and train_losses and eval_epochs and eval_losses:
            # Scatter plot might become too cluttered with multiple lines
            # plt.scatter(train_epochs, train_losses, alpha=0.1, color=color, s=5) # , label=f'{label_prefix} Raw Loss'

            # Calculate Moving Average
            averages = []
            window_size = min(WINDOW, len(train_losses))
            for i in range(len(train_losses)):
                window = train_losses[max(0, i - window_size + 1) : i + 1] # Corrected window slicing
                averages.append(sum(window) / len(window))

            if PLOT_TRAINING:
                plt.plot(train_epochs, averages, color=color, label=f'{label_prefix} avg train loss step {checkpoint_num}')
                plt.axhline(y=min(averages), linestyle=':', alpha=0.6, color=color)

            if PLOT_EVALUATION:
                # --- Find Checkpoints (Multiples of 500) ---
                checkpoint_epochs, checkpoint_losses, checkpoint_steps = [], [], []
                for e, l, s in zip(eval_epochs, eval_losses, eval_steps):
                    if s % 500 == 0:
                        checkpoint_epochs.append(e)
                        checkpoint_losses.append(l)
                        checkpoint_steps.append(s) # Store the step number as well

                # --- Plotting ---
                # Plot scatter points for the checkpoints found
                if checkpoint_epochs: # Only plot if there are checkpoints
                    plt.scatter(checkpoint_epochs, checkpoint_losses, s=25, color=color, marker='o', linewidth=0.5, label=f'{label_prefix} evaluation loss checkpoint') # Make markers stand out a bit

                # Plot the main evaluation loss line
                plt.plot(eval_epochs, eval_losses, alpha=1.0, color=color)
                plt.axhline(y=min(eval_losses), linestyle=':', alpha=0.6, color=color)

                # --- Annotation Logic Start ---
                if checkpoint_losses: # Ensure we have checkpoint data to find a minimum
                    # Find the minimum loss among the *checkpoints*
                    min_val = min(checkpoint_losses)
                    min_idx = checkpoint_losses.index(min_val)

                    # Add the annotation to the plot
                    plt.annotate(f'v{version}, loss {min_val:.4}, step {checkpoint_steps[min_idx]}', # The text to display
                                 xy=(checkpoint_epochs[min_idx], min_val), # The point (epoch, loss) to annotate
                                 xytext=(10, 10), # Offset the text slightly
                                 textcoords='offset points', # Use offset points for positioning
                                 ha='left',         # Horizontal alignment
                                 va='bottom',       # Vertical alignment
                                 arrowprops=dict(arrowstyle='->', lw=0.8, connectionstyle="arc3,rad=0.2"), # Arrow properties
                                 color=color,       # Text color matching the line
                                 fontsize=9,        # Adjust font size if needed
                                 bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.6)) # Optional background box
                # --- Annotation Logic End ---

    print("-" * 20)

    # Configure plot after processing all directories
    # plt.xlim(0, 2.0) # Keep commented unless needed
    plt.ylim(2.5, 3.2) # Keep existing limit or adjust as needed

    # Add overall labels and styling
    plt.title(f'Training & Evaluation Loss Comparison (Min Checkpoint Annotated)') # Updated title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.4, linestyle=':')
    plt.legend() # Display the legend with labels for each directory/line
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()