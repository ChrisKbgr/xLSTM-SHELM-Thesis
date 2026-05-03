import wandb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Initialize W&B API
api = wandb.Api()

# Fetch the three runs
run_ids = ["aynv7mg3", "zpsv9osm", "qo35lzqs"]
project = "chr-kuehberger-johannes-kepler-universit-t-linz/shelm-minigrid-memory-xlstm-final"

runs_data = {}
for run_id in run_ids:
    run = api.run(f"{project}/{run_id}")
    runs_data[run_id] = run.history()
    print(f"Loaded run {run_id}: {len(runs_data[run_id])} steps")

print("\nAvailable metrics:")
print(runs_data[run_ids[0]].columns.tolist())

# Example 1: Simple line plots
def plot_training_curves(df, metric_name, title=None, output_file=None):
    """Plot a single metric over time (e.g., loss, accuracy)"""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[metric_name], linewidth=2)
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.title(title or f"{metric_name} over time")
    plt.grid(True, alpha=0.3)
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

# Example 2: Multiple metrics on same plot
def plot_multiple_metrics(df, metric_names, title=None, output_file=None):
    """Plot multiple metrics on the same figure"""
    plt.figure(figsize=(12, 6))
    for metric in metric_names:
        if metric in df.columns:
            plt.plot(df.index, df[metric], label=metric, linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title or "Training Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

# Example 3: Subplots for different metric groups
def plot_subplots(df, metric_groups, figsize=(15, 10), output_file=None):
    """Create a subplot grid for different metric groups"""
    n = len(metric_groups)
    fig, axes = plt.subplots(n, 1, figsize=figsize)
    
    for idx, (title, metrics) in enumerate(metric_groups.items()):
        ax = axes[idx] if n > 1 else axes
        for metric in metrics:
            if metric in df.columns:
                ax.plot(df.index, df[metric], label=metric, linewidth=2)
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Step") if n > 1 else axes.set_xlabel("Step")
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

# USAGE EXAMPLES (uncomment to use):
# =====================================

# Plot reward vs total timesteps for all 3 runs
fig, ax = plt.subplots(figsize=(12, 7))

# Set professional style
ax.set_facecolor("#e5eef1")  # Light grey background
fig.patch.set_facecolor('white')

# Format x-axis to show K and M (thousands and millions)
def format_func(value, tick_number):
    return f'{value/1e6:.1f}M'


ax.xaxis.set_major_formatter(FuncFormatter(format_func))
# Set ticks at 0.2M intervals (200,000 steps)
ax.xaxis.set_major_locator(MultipleLocator(200000))

colors = ["#3a7aa8", "#D1DA57", "#33a833"]  # Different colors for each run
names  = ['XLSTM Layernorm Standard lr', 'XLSTM Layernorm lr=0.00005', 'XLSTM lr=0.00005']  # Legend labels

# Track min/max values for tight axis limits
x_min, x_max = float('inf'), 0
y_min, y_max = float('inf'), 0

for idx, run_id in enumerate(run_ids):
    df = runs_data[run_id]
    
    # Use total_timesteps as x-axis and episode reward mean as y-axis
    x = df['time/total_timesteps'].dropna()
    y = df['rollout/ep_rew_mean'].dropna()
    
    # Align the two series (in case they have different lengths due to NaN values)
    aligned_data = pd.DataFrame({'x': x, 'y': y}).dropna()
    
    # Track bounds
    x_min = min(x_min, aligned_data['x'].min())
    x_max = max(x_max, aligned_data['x'].max())
    y_min = min(y_min, aligned_data['y'].min())
    y_max = max(y_max, aligned_data['y'].max())
    
    ax.plot(aligned_data['x'], aligned_data['y'], 
            label=f'{names[idx]}', linewidth=2.5, color=colors[idx])

# Set tight axis limits with small margins
x_margin = (x_max - x_min) * 0.02
y_margin = (y_max - y_min) * 0.02
ax.set_xlim(x_min - x_margin, x_max + x_margin)
ax.set_ylim(y_min - y_margin, y_max + y_margin*3)  # Add extra vertical margin for legend

# Add white grid lines
ax.grid(True, color='white', linewidth=1.2, alpha=0.8, linestyle='-', zorder=0)
ax.set_axisbelow(True)

# Remove all spines (black outline)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_xlabel('Number of Interaction Steps', fontsize=12)
ax.set_ylabel('Accumulated Reward', fontsize=12)
ax.set_title('MiniGrid - Memory', fontsize=14)
ax.legend(fontsize=11, framealpha=0)

plt.tight_layout()
plt.savefig('reward_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()