import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import io
import torchvision.transforms.functional as TF
from PIL import Image
import logging

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
