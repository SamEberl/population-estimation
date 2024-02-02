import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import io
import torchvision.transforms.functional as TF
from PIL import Image
import logging
import numpy as np
import os
import torch

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


# def plot_uncertainty(teacher_mean_flat, teacher_var_flat, teacher_loss_flat, actual_label_flat, uncertainties_flat):
def plot_uncertainty(teacher_mean, teacher_var, teacher_loss, actual_label, uncertainties, euclidean_spread):
    # Convert tensors in the lists to flat numpy arrays
    teacher_mean_numpy = [v.cpu().numpy().flatten() for v in teacher_mean]
    teacher_var_numpy = [v.cpu().numpy().flatten() for v in teacher_var]
    teacher_loss_numpy = [v.cpu().numpy().flatten() for v in teacher_loss]
    actual_label_numpy = [v.cpu().numpy().flatten() for v in actual_label]
    uncertainties_numpy = [v.cpu().detach().numpy().flatten() for v in uncertainties]
    euclidean_spread_numpy = [v.cpu().detach().numpy().flatten() for v in euclidean_spread]

    # Ensure all arrays are flat and concatenated
    teacher_mean_flat = np.concatenate(teacher_mean_numpy)
    teacher_var_flat = np.concatenate(teacher_var_numpy)
    teacher_loss_flat = np.concatenate(teacher_loss_numpy)
    actual_label_flat = np.concatenate(actual_label_numpy)
    uncertainties_flat = np.concatenate(uncertainties_numpy)
    euclidean_spread_flat = np.concatenate(euclidean_spread_numpy)

    # path = '/home/sam/Desktop/logs/figs/'
    path = '/home/sameberl/logs/computed_numpy'

    np.save(os.path.join(path, 'teacher_mean_flat.npy'), teacher_mean_flat)
    np.save(os.path.join(path, 'teacher_var_flat.npy'), teacher_var_flat)
    np.save(os.path.join(path, 'teacher_loss_flat.npy'), teacher_loss_flat)
    np.save(os.path.join(path, 'actual_label_flat.npy'), actual_label_flat)
    np.save(os.path.join(path, 'uncertainties_flat.npy'), uncertainties_flat)
    np.save(os.path.join(path, 'euclidean_spread_flat.npy'), euclidean_spread_flat)

    # teacher_loss_flat = teacher_loss_flat / (actual_label_flat + 1)
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
    ax2.scatter(uncertainties_flat, teacher_loss_flat, color='red', s=s)
    ax2.set_title("Uncertainty vs. Loss")
    ax2.set_xlabel("Uncertainty")
    ax2.set_ylabel("Loss")
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('/home/sam/Desktop/logs/figs/measureOverLoss1.png')

