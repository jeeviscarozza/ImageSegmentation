import argparse
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from data import get_dataset
from utils import get_training_args
from sklearn.metrics import confusion_matrix
from pathlib import Path
import os
from models import UNet
from train import weighted_binary_crossentropy, compute_iou

# Images are evaluated in the training script, since there is an issue loading the model when using custom metrics

def eval(args):
    # Load dataset
    test_data = get_dataset(
        batch_size=args.batch_size,
        img_size=(128, 128),
        input_img_paths=Path(args.input_img_paths).glob('test/images/*.png'),
        target_img_paths=Path(args.target_img_paths).glob('test/masks/*_mask.png'),
    )
    print("testing")

    # # For each model load the args used during training
    # model_args = []
    # for model_name in args.model_names:
    #     print(model_name)
    #     file_name = f"{args.model_dir}/{model_name}/args.yaml"
    #     with open(file_name) as f:
    #         model_args.append(yaml.safe_load(f))

    # # Pull out the important args
    # important_args = {
    #     a: [model_args[i][a] for i in range(len(args.model_names))] for a in args.important_args
    # }

    # Initialize lists to store accuracy and IOU for each model
    accuracies = []
    ious = []

    # Evaluate each model
    for model_name in args.model_names:
        print(model_name)
        model = tf.keras.models.load_model(f"{args.model_dir}/{model_name}/{model_name}.keras",compile=False)
        y_true_sample = []
        print("hi")
        y_pred_sample = []
        for input_img, target_mask in test_data:
            y_true_sample.extend(np.argmax(target_mask, axis=-1).flatten())
            predictions = model.predict(input_img)
            y_pred_sample.extend(np.argmax(predictions, axis=-1).flatten())
        
        # Calculate accuracy
        acc = np.mean(np.array(y_true_sample) == np.array(y_pred_sample))
        accuracies.append(acc)

        # Calculate IOU
        intersection = np.logical_and(np.array(y_true_sample), np.array(y_pred_sample))
        union = np.logical_or(np.array(y_true_sample), np.array(y_pred_sample))
        iou = np.sum(intersection) / np.sum(union)
        ious.append(iou)

        # Save sample image, mask, and prediction
        for idx, (input_img, target_mask) in enumerate(test_data.take(1)):
            predicted_mask = model.predict(input_img)
            display_evaluate_image(input_img[0], predicted_mask[0], target_mask[0], idx, output_dir)

    # extracted_layers = {"Dropout": [], "Activation": []}

    # # Create a table showing: model name, important args, accuracy, IOU
    # df = pd.DataFrame()
    # df["Model"] = args.model_names
    # for layer in model.layers:
    #     if isinstance(layer, tf.keras.layers.Dropout):
    #         extracted_layers["Dropout"].append(layer)
    #     elif isinstance(layer, tf.keras.layers.Activation):
    #         extracted_layers["Activation"].append(layer.activation)
    # df["Test Accuracy"] = accuracies
    # df["Test IOU"] = ious

    model_instances = []
    for model_name in args.model_names:
        model_path = f"{args.model_dir}/{model_name}/{model_name}.keras"
        model = tf.keras.models.load_model(model_path, custom_objects={'UNet': UNet, 'loss_fn': weighted_binary_crossentropy, 'compute_iou': compute_iou})
        model_instances.append(model)

    extracted_layers = {"Dropout": [], "Activation": []}

    history = model.history.history if hasattr(model, 'history') else None
    if history:
        print(f"History keys for model {model_name}: {history.keys()}")
        plt.plot(history['accuracy'], label='accuracy')
        plt.plot(history['val_accuracy'], label='val_accuracy')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    for model in model_instances:
        dropout_values = []
        activation_values = []
        
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                dropout_values.append(layer.rate)
            elif isinstance(layer, tf.keras.layers.Activation):
                activation_values.append(layer.activation.__name__)
        
        extracted_layers["Dropout"].append(dropout_values[0] if dropout_values else None)
        extracted_layers["Activation"].append(activation_values[0] if activation_values else None)

    df = pd.DataFrame()
    df["Model"] = args.model_names
    df["Dropout"] = extracted_layers["Dropout"]
    df["Activation"] = extracted_layers["Activation"]
    df["Test Accuracy"] = accuracies
    df["Test IOU"] = ious

    print(df)


    # Create a DataFrame
    df = pd.DataFrame()
    df["Model"] = args.model_names
    df["Dropout"] = extracted_layers["Dropout"]
    df["Activation"] = extracted_layers["Activation"]
    df["Test Accuracy"] = accuracies
    df["Test IOU"] = ious

    print(df)

    # Create output directory for comparison results
    comparison_dir = Path("comparison_results")
    comparison_dir.mkdir(exist_ok=True, parents=True)

    # Save results as Markdown table
    df.to_markdown(comparison_dir / "accuracy_iou.md", index=False)

    # Plot the validation accuracy and IOU curves
    fig, ax = plt.subplots()
    for model_name in args.model_names:
        print(history)
        history = pd.read_csv(f"{args.model_dir}/{model_name}/history.csv")
        ax.plot(history["val_accuracy"], label=f"{model_name} - Accuracy")
        ax.plot(history["val_iou"], label=f"{model_name} - IOU")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(comparison_dir / "val_metrics.png")
    plt.close()

# def display_evaluate_image(input_image, predicted_mask, target_mask, image_index, output_dir):
#     """
#     Displays the  input image, predicted mask, and target mask, and saves them to the output directory.
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     axes[0].imshow(tf.squeeze(input_image))
#     axes[0].set_title("Input Image")
#     axes[0].axis("off")

#     axes[1].imshow(tf.squeeze(predicted_mask), cmap="gray")
#     axes[1].set_title("Predicted Mask")
#     axes[1].axis("off")

#     axes[2].imshow(tf.squeeze(target_mask), cmap="gray")
#     axes[2].set_title("Target Mask")
#     axes[2].axis("off")

#     # Save the figure
#     fig.savefig(os.path.join(output_dir, f"model_results{image_index}.png"))
#     plt.close(fig)

#     # Print the index of the input image
#     print("Image index:", image_index)