import matplotlib.pyplot as plt

def log_images_scores_to_tensorboard(writer, step_nbr, sample_nbr, input, label, prediction):
    # Log input image as a grid to TensorBoard
    # writer.add_images(f'Input_Image_{sample_nbr}', input, global_step=step_nbr, dataformats='CHW')
    writer.add_images(f'Images_Panel', input, global_step=step_nbr, dataformats='CHW')

    # Log actual and predicted scores as scalars
    # writer.add_scalar(f'Actual_Score_{sample_nbr}', label, global_step=step_nbr)
    # writer.add_scalar(f'Predicted_Score_{sample_nbr}', prediction, global_step=step_nbr)



def log_regression_plot_to_tensorboard(writer, step_nbr, labels, predictions):
    fig, ax = plt.subplots()
    ax.scatter(labels, predictions)
    ax.plot([min(labels), max(predictions)], [min(labels), max(predictions)], linestyle='--', color='r')
    ax.set_xlabel('Actual Score')
    ax.set_ylabel('Predicted Score')
    ax.set_title(f'Regression Plot - Epoch {step_nbr}')

    writer.add_figure('Regression_Plot', fig, global_step=step_nbr)
    plt.close()
