import matplotlib.pyplot as plt
import numpy as np
import config

def plot_zeroshot_vs_linear(zs_results, lp_results):
    datasets = sorted(lp_results.keys())
    zs_scores = [zs_results[d]['accuracy'] for d in datasets if d in zs_results]
    lp_scores = [lp_results[d]['accuracy'] for d in datasets if d in lp_results]
    
    if not zs_scores or not lp_scores:
        print("Insufficient data for Plot A")
        return

    labels = datasets
    y = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y - width/2, zs_scores, width, label='Zero-Shot', color='blue')
    ax.barh(y + width/2, lp_scores, width, label='Linear-Probe', color='orange')
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Zero-Shot vs. Linear-Probe')
    ax.legend()
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(config.RESULTS_FOLDER / "zeroshot_vs_linear.png")
    print("Plot saved.")

def plot_few_shot(zs_results, fs_results):
    # (Implementation of Plot C from original code, adapted for new structure)
    pass 
    # You can copy paste the logic from your notebook here if needed, 
    # accessing constants from config.py