import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import io
import torchvision.transforms.functional as TF
from PIL import Image
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_images_scores_to_tensorboard(writer, step_nbr, input):
    # Log input image as a grid to TensorBoard
    writer.add_images(f'Panel - Images', input, global_step=step_nbr, dataformats='CHW')

# def log_regression_plot_to_tensorboard(writer, step_nbr, labels, predictions):
#     fig, ax = plt.subplots()
#     ax.scatter(labels, predictions)
#     ax.plot([min(labels), max(predictions)], [min(labels), max(predictions)], linestyle='--', color='r')
#     ax.set_xlabel('Actual Score')
#     ax.set_ylabel('Predicted Score')
#     ax.set_title(f'Regression Plot - Epoch {step_nbr}')
#
#     writer.add_figure('Regression_Plot', fig, global_step=step_nbr)
#     plt.close()


def log_regression_plot_to_tensorboard(writer, step_nbr, labels, predictions):
    fig, ax = plt.subplots()

    ax.scatter(labels, predictions)
    ax.plot([min(labels), max(predictions)], [min(labels), max(predictions)], linestyle='--', color='r')

    ax.set_xscale('log')  # Set x-axis to logarithmic scale
    ax.set_yscale('log')  # Set y-axis to logarithmic scale

    ax.set_xlabel('Actual Score')
    ax.set_ylabel('Predicted Score')
    ax.set_title(f'Logarithmic Regression Plot - Epoch {step_nbr}')

    writer.add_figure('Panel_Logarithmic_Regression_Plot', fig, global_step=step_nbr)
    plt.close()


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.title('Confusion Matrix')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf


def log_confusion_matrix_to_tensorboard(writer, step_nbr, labels, predictions, class_names):
    cm = confusion_matrix(labels, predictions)
    cm_image_buf = plot_confusion_matrix(cm, class_names)

    # Convert the image buffer to a PyTorch tensor
    cm_image_tensor = TF.to_tensor(Image.open(cm_image_buf))

    # Add the confusion matrix image as a grid to TensorBoard
    writer.add_image('Confusion_Matrix', cm_image_tensor, global_step=step_nbr, dataformats='CHW')


def plot_uncertainty(teacher_mean, teacher_var, teacher_loss, actual_label, uncertainties):
    # Convert tensors in the lists to flat numpy arrays
    teacher_mean_numpy = [v.cpu().numpy().flatten() for v in teacher_mean]
    teacher_var_numpy = [v.cpu().numpy().flatten() for v in teacher_var]
    teacher_loss_numpy = [v.cpu().numpy().flatten() for v in teacher_loss]
    uncertainties_numpy = [v.cpu().detach().numpy().flatten() for v in uncertainties]

    # Ensure all arrays are flat and concatenated
    teacher_mean_flat = np.concatenate(teacher_mean_numpy)
    teacher_var_flat = np.concatenate(teacher_var_numpy)
    teacher_loss_flat = np.concatenate(teacher_loss_numpy)
    teacher_loss_flat = np.sqrt(teacher_loss_flat)+1 / (actual_label+1)
    uncertainties_flat =np.concatenate(uncertainties_numpy)

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
    ax2.scatter(uncertainties_flat, teacher_loss_flat, color='red', s=s)
    ax2.set_title("SNR vs. Loss")
    ax2.set_xlabel("SNR")
    ax2.set_ylabel("Loss")
    # ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('/home/sam/Desktop/logs/figs/measureOverLoss.png')


def plot_uncertainty_histograms(teacher_mean, teacher_var):
    # Convert tensors in the lists to flat numpy arrays
    teacher_mean_numpy = [v.cpu().numpy().flatten() for v in teacher_mean]
    teacher_var_numpy = [v.cpu().numpy().flatten() for v in teacher_var]

    # Ensure all arrays are flat and concatenated
    teacher_mean_flat = np.concatenate(teacher_mean_numpy)
    teacher_var_flat = np.concatenate(teacher_var_numpy)

    bins = 80

    # Histogram for Mean
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(teacher_mean_flat, bins=bins, color='blue', edgecolor='black')
    ax1.set_title("Histogram of Mean")
    ax1.set_xlabel("Mean")
    ax1.set_ylabel("Frequency")
    # ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Histogram for Variance
    ax2 = plt.subplot(2, 2, 2)
    ax2.hist(teacher_var_flat, bins=bins, color='red', edgecolor='black')
    ax2.set_title("Histogram of Variance")
    ax2.set_xlabel("Variance")
    ax2.set_ylabel("Frequency")
    # ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Histogram for Fano Factor (Var/Mean)
    ax4 = plt.subplot(2, 2, 3)
    ax4.hist(teacher_var_flat / teacher_mean_flat, bins=bins, color='yellow', edgecolor='black')
    ax4.set_title("Histogram of Fano Factor")
    ax4.set_xlabel("Var/Mean")
    ax4.set_ylabel("Frequency")
    # ax4.set_xscale('log')
    ax4.set_yscale('log')

    # Histogram for Coefficient of Variation (CV)
    ax5 = plt.subplot(2, 2, 4)
    ax5.hist(np.sqrt(teacher_var_flat) / teacher_mean_flat, bins=bins, color='green', edgecolor='black')
    ax5.set_title("Histogram of CV")
    ax5.set_xlabel("CV")
    ax5.set_ylabel("Frequency")
    # ax5.set_xscale('log')
    # ax5.set_yscale('log')

    # # Histogram for Signal-to-Noise Ratio (SNR)
    # ax6 = plt.subplot(3, 3, 6)
    # ax6.hist(teacher_mean_flat ** 2 / teacher_var_flat, bins=bins, color='green', edgecolor='black')
    # ax6.set_title("Histogram of SNR")
    # ax6.set_xlabel("SNR")
    # ax6.set_ylabel("Frequency")
    # # ax6.set_xscale('log')
    # # ax6.set_yscale('log')

    plt.tight_layout()
    plt.savefig('/home/sam/Desktop/logs/figs/histograms.png')