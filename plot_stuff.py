import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


plt.rcParams['font.size'] = 12
# plt.rcParams['font.family'] = 'Helvetica'


def plot_uncertainty(teacher_mean_flat, teacher_var_flat, teacher_loss_flat, actual_label_flat, uncertainties_flat):
    # teacher_loss_flat = teacher_loss_flat / (actual_label_flat + 1)
    teacher_loss_flat = np.sqrt(teacher_loss_flat) / (actual_label_flat+1)
    # teacher_loss_flat = actual_label_flat

    s = 1

    # Plot for mean vs. loss
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(2, 3, 1)  # 1 row, 2 columns, 1st plot
    ax1.scatter(teacher_mean_flat, teacher_loss_flat, color='blue', s=s)
    ax1.set_title("Mean vs. Loss")
    ax1.set_xlabel("Mean")
    ax1.set_ylabel("Loss")
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Plot for var vs. loss
    ax2 = plt.subplot(2, 3, 2)  # 1 row, 2 columns, 2nd plot
    ax2.scatter(teacher_var_flat, teacher_loss_flat, color='red', s=s)
    ax2.set_title("Var vs. Loss")
    ax2.set_xlabel("Var")
    ax2.set_ylabel("Loss")
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2 = plt.subplot(2, 3, 3)  # 1 row, 2 columns, 2nd plot
    ax2.scatter(teacher_var_flat/teacher_mean_flat, teacher_loss_flat, color='black', s=s)
    ax2.set_title("Fano vs. Loss")
    ax2.set_xlabel("Var/Mean")
    ax2.set_ylabel("Loss")
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2 = plt.subplot(2, 3, 4)  # 1 row, 2 columns, 2nd plot
    ax2.scatter(np.sqrt(teacher_var_flat)/teacher_mean_flat, teacher_loss_flat, color='green', s=s)
    ax2.set_title("CV vs. Loss")
    ax2.set_xlabel("CV")
    ax2.set_ylabel("Loss")
    # ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2 = plt.subplot(2, 3, 5)  # 1 row, 2 columns, 2nd plot
    ax2.scatter(teacher_mean_flat**2/teacher_var_flat, teacher_loss_flat, color='purple', s=s)
    ax2.set_title("SNR vs. Loss")
    ax2.set_xlabel("SNR")
    ax2.set_ylabel("Loss")
    # ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2 = plt.subplot(2, 3, 6)  # 1 row, 2 columns, 2nd plot
    ax2.scatter(np.sqrt(uncertainties_flat)/teacher_mean_flat, teacher_loss_flat, color='red', s=s)
    ax2.set_title("Uncertainty vs. Loss")
    ax2.set_xlabel("Uncertainty")
    ax2.set_ylabel("Loss")
    ax2.set_xscale('log')
    # ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('/home/sam/Desktop/logs/figs/measureOverLoss2.png')


def plot_uncertainty_histograms(teacher_mean, teacher_var, uncertainties_flat):
    bins = 80

    # Histogram for Mean
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(teacher_mean_flat, bins=bins, color='blue', edgecolor='black')
    ax1.set_title("Histogram of Mean")
    ax1.set_xlabel("Mean")
    ax1.set_ylabel("Frequency")
    # ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Histogram for Variance
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(teacher_var_flat, bins=bins, color='red', edgecolor='black')
    ax2.set_title("Histogram of Variance")
    ax2.set_xlabel("Variance")
    ax2.set_ylabel("Frequency")
    # ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Histogram for Fano Factor (Var/Mean)
    ax4 = plt.subplot(2, 3, 3)
    ax4.hist(teacher_var_flat / teacher_mean_flat, bins=bins, color='yellow', edgecolor='black')
    ax4.set_title("Histogram of Fano Factor")
    ax4.set_xlabel("Var/Mean")
    ax4.set_ylabel("Frequency")
    # ax4.set_xscale('log')
    ax4.set_yscale('log')

    # Histogram for Coefficient of Variation (CV)
    ax5 = plt.subplot(2, 3, 4)
    ax5.hist(np.sqrt(teacher_var_flat) / teacher_mean_flat, bins=bins, color='green', edgecolor='black')
    ax5.set_title("Histogram of CV")
    ax5.set_xlabel("CV")
    ax5.set_ylabel("Frequency")
    # ax5.set_xscale('log')
    # ax5.set_yscale('log')

    # Histogram for Signal-to-Noise Ratio (SNR)
    ax6 = plt.subplot(2, 3, 5)
    ax6.hist(teacher_mean_flat ** 2 / teacher_var_flat, bins=bins, color='green', edgecolor='black')
    ax6.set_title("Histogram of SNR")
    ax6.set_xlabel("SNR")
    ax6.set_ylabel("Frequency")
    # ax6.set_xscale('log')
    # ax6.set_yscale('log')

    ax3 = plt.subplot(2, 3, 6)
    # ax3.hist(np.sqrt(uncertainties_flat)/teacher_mean_flat, bins=bins, color='green', edgecolor='red')
    ax3.hist(teacher_mean_flat**2/uncertainties_flat, bins=bins, color='green', edgecolor='red')
    ax3.set_title("Uncertainty vs. Loss")
    ax3.set_xlabel("Uncertainty")
    ax3.set_ylabel("Loss")
    # ax3.set_xscale('log')
    # ax3.set_yscale('log')

    plt.tight_layout()
    plt.savefig('/home/sam/Desktop/logs/figs/histograms.png')


def plot_percentiles():
    nbr_channels = 20

    histogram_dir = Path("/home/sam/Desktop/so2sat_test/histograms")
    # After value_counts has been computed
    value_counts_file = histogram_dir/'value_counts.pkl'

    with open(value_counts_file, 'rb') as f:
        value_counts = pickle.load(f)

    use_log_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # use_log_y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Define color as RGB tuple
    tum_blue = '#0065bd'

    # Create and save histograms
    for c in tqdm(range(nbr_channels)):
        values = list(value_counts[c].keys())
        counts = list(value_counts[c].values())

        # Determine bin width based on the range of values
        min_value, max_value = min(values), max(values)
        range_of_values = max_value - min_value
        bin_width = range_of_values / 100  # for example, create 50 bins

        plt.figure()
        plt.hist(values, bins=np.arange(min_value, max_value + bin_width, bin_width), weights=counts, color=tum_blue)

        # Apply logarithmic scale if specified
        if use_log_y[c] == 1:
            plt.yscale('log')

        plt.title(f'Histogram for Channel {c+1}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        if use_log_y[c] == 1:
            plt.savefig(f'{histogram_dir}/log_channel_{c+1}_sen2spring.png')
        else:
            plt.savefig(f'{histogram_dir}/channel_{c+1}_sen2spring.png')
        plt.close()




def visualize_features(features, labels=None, method='pca', save_path='feature_visualization.png', perplexity=30, n_iter=300):
    """
    Visualize high-dimensional feature vectors using PCA or t-SNE.

    Parameters:
    features (np.array): High-dimensional data to visualize.
    labels (np.array): Labels for the data points, if available. Used for color coding.
    method (str): Visualization method - 'pca' or 'tsne'.
    save_path (str): Path to save the visualization image.
    perplexity (int): Perplexity for t-SNE. Usually between 5 and 50.
    n_iter (int): Number of iterations for t-SNE optimization.
    """
    if method == 'pca':
        reducer = PCA(n_components=2)
        title = 'PCA of Features'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
        title = 't-SNE of Features'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    reduced_features = reducer.fit_transform(features)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.5)
    else:
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.5)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    plt.colorbar() if labels is not None else None
    plt.savefig(save_path)
    plt.close()

    return save_path

# Example usage:
import pandas as pd
# features = pd.read_csv('/home/sam/Desktop/train_features_full.csv')
features = pd.read_csv('/home/sam/Desktop/val_features_sen2spring_full.csv')
print(features.shape)
features_val = features.iloc[:, :-1]
labels = features.iloc[:, -1]
visualize_features(features, labels=labels, method='tsne', save_path='/home/sam/Desktop/so2sat_test/tsne_val_visualization.png')
# Note: Replace 'your_feature_array' and 'your_labels' with your actual data.


# plot_percentiles()


# teacher_mean_flat = np.load('/home/sam/Desktop/logs/figs/computed_numpy/teacher_mean_flat.npy')
# teacher_var_flat = np.load('/home/sam/Desktop/logs/figs/computed_numpy/teacher_var_flat.npy')
# teacher_loss_flat = np.load('/home/sam/Desktop/logs/figs/computed_numpy/teacher_loss_flat.npy')
# actual_label_flat = np.load('/home/sam/Desktop/logs/figs/computed_numpy/actual_label_flat.npy')
# uncertainties_flat = np.load('/home/sam/Desktop/logs/figs/computed_numpy/uncertainties_flat.npy')
#
# plot_uncertainty(teacher_mean_flat, teacher_var_flat, teacher_loss_flat, actual_label_flat, uncertainties_flat)
# plot_uncertainty_histograms(teacher_mean_flat, teacher_var_flat, uncertainties_flat)