"""A collection of utility functions that are used in the other files.
They are organised here to make the other files easier to read.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf


def get_training_args() -> argparse.Namespace:
    """Get the arguments needed for the training script."""
    # This is the standard way to define command line arguments in python,
    # without using config files which you are welcome to do!
    # Here the argparse module can pick up command line arguments and return them
    # So if you type in `python train.py --num_mlp_layers 3` it will save the value 3
    # to `args.num_mlp_layers`.
    # This allows you to change the model without changing the code!
    # Each possible argument must be defined here.
    # Feel free to add more arguments as you see fit.

    # First we have to create a parser object
    parser = argparse.ArgumentParser()

    # Define the important paths for the project
    parser.add_argument(
        "--model_dir",  # How we access the argument when calling `python train.py ...`
        type=str,  # We must also define the type of argument, here it is a string
        default="models",  # The default value so you dont have to type it in every time
        help="Where to save trained models",  # A helpfull message
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="UNet",
        help="The name of the model",
    )

    # Arguments for the network
    parser.add_argument(
        "--num_filters_per_layer",
        type=str,
        default="8,16,32",
        help="A comma separated list of the number of filters per layer",
    )
    parser.add_argument(
        "--num_mlp_layers",
        type=int,
        default=2,
        help="The number of MLP layers",
    )
    parser.add_argument(
        "--num_mlp_hidden",
        type=int,
        default=32,
        help="The number of hidden units",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="The activation function, see https://keras.io/api/layers/activations/",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="The dropout rate",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="batch",
        help="The normalization type",
    )

    # Arguments for how to train the model
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="mse",
        help="The loss function, see https://keras.io/api/losses/",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="The maximum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=30,
        help="The maximum number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="The learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        help="The weight decay",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="The optimizer, see https://keras.io/api/optimizers/",
    )
    parser.add_argument(
        "--input_img_paths",
        type=Path,
        default="preprocessed_data",
        help="The path to the input images",
    )
    parser.add_argument(
        "--target_img_paths",
        type=Path,
        default="preprocessed_data",
        help="The path to the target images",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model",
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the model",
    )
    parser.add_argument(
        "--image_index", 
        type=int,
        help="Index of the image to evaluate."
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=["unet_1","unet_1_lr","unet_batch_128"],
        help="A comma separated list of model names to load and compare",
    )
    parser.add_argument(
        "--important_args",
        type=str,
        nargs="+",
        default=["loss_fn","dropout","activation","num_filters_per_layer"],
        help="A comma separated list of args to include in the plots",
    )  

    # This now collects all arguments
    args = parser.parse_args()

    # We have to do some extra work to parse the list of integers
    args.num_filters_per_layer = [int(x) for x in args.num_filters_per_layer.split(",")]

    # Now we return the arguments
    return args



def plot_history(history, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 22))

    axes[0, 0].plot(history.history["loss"], label="Train Loss")
    axes[0, 0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(history.history["binary_accuracy"], label="Train Accuracy")
    axes[0, 1].plot(history.history["val_binary_accuracy"], label="Validation Accuracy")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(history.history["precision"], label="Train Precision")
    axes[1, 0].plot(history.history["val_precision"], label="Validation Precision")
    axes[1, 0].plot(history.history["recall"], label="Train Recall")
    axes[1, 0].plot(history.history["val_recall"], label="Validation Recall")
    axes[1, 0].set_title("Precision and Recall")
    axes[1, 0].legend()
    
    axes[1, 1].plot(history.history["compute_iou"], label="Train IoU")
    axes[1, 1].plot(history.history["val_compute_iou"], label="Validation IoU")
    axes[1, 1].set_title("IoU")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

