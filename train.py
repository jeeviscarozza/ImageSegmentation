from data import image_masks
import tensorflow as tf
from data import get_dataset
from utils import get_training_args
from utils import plot_history
from models import UNet
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2
#from visualize_results import visualize_results

args = get_training_args()
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pathlib import Path

def visualize_results(model, test_data, output_dir, img_size=(128, 128)):
    for idx, (input_image, target_mask) in enumerate(test_data.take(4)):
        # input_image and target_mask are already processed by get_dataset, so no need for additional resizing and normalization
        input_image = tf.image.resize(input_image[0], img_size) 
        target_mask = tf.image.resize(target_mask[0], img_size)  

        input_image = tf.expand_dims(input_image, axis=0)  # Add batch dimension

        predicted_mask = model.predict(input_image)
        predicted_mask = tf.round(predicted_mask[0])  # Remove batch dimension and round the prediction

        plt.figure(figsize=(15, 5))

        # Plot and save input image
        plt.subplot(1, 3, 1)
        plt.imshow(input_image[0])
        plt.title(f"Input Image {idx+1}")
        plt.axis("off")
        input_image_name = f"input_image_{idx+1}.png"
        input_image_path = output_dir / "visualized_results" / input_image_name
        input_image_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(input_image_path)

        # Plot and save target mask
        plt.subplot(1, 3, 2)
        plt.imshow(tf.squeeze(target_mask), cmap='gray')
        plt.title(f"Target Mask {idx+1}")
        plt.axis("off")
        target_mask_name = f"target_mask_{idx+1}.png"
        target_mask_path = output_dir / "visualized_results" / target_mask_name
        plt.savefig(target_mask_path)

        # Plot and save predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(tf.squeeze(predicted_mask), cmap='gray')
        plt.title(f"Predicted Mask {idx+1}")
        plt.axis("off")
        predicted_mask_name = f"predicted_mask_{idx+1}.png"
        predicted_mask_path = output_dir / "visualized_results" / predicted_mask_name
        plt.savefig(predicted_mask_path)

        plt.show()
        plt.close()

#custom loss function
@tf.keras.utils.register_keras_serializable()
def weighted_binary_crossentropy(class_weights):
    def loss_fn(y_true, y_pred):
        # Flatten the tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Calculate the binary cross-entropy loss
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Apply class weights
        weight_vector = y_true * class_weights[1] + (1 - y_true) * class_weights[0]
        weighted_bce = weight_vector * bce
        
        return tf.reduce_mean(weighted_bce)
    return loss_fn

#Computation for iou
@tf.keras.utils.register_keras_serializable()
def compute_iou(y_true, y_pred, threshold=0.5):
    y_true = tf.greater(y_true, threshold)
    y_pred = tf.greater(y_pred, threshold)
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(y_true, y_pred), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.logical_or(y_true, y_pred), tf.float32))
    iou = tf.where(union > 0, intersection / union, 0)
    return iou


def train(args):

    # Load and prepare input and target images
    train_data = get_dataset(
        batch_size=args.batch_size,
        img_size=(128, 128),
        input_img_paths=(args.input_img_paths/'train/images').glob('*.png'),
        target_img_paths=(args.target_img_paths/'train/masks').glob('*_mask.png'),
    )
    val_data = get_dataset(
        batch_size=args.batch_size,
        img_size=(128, 128),
        input_img_paths=(args.input_img_paths/'val/images').glob('*.png'),
        target_img_paths=(args.target_img_paths/'val/masks').glob('*_mask.png'),
    )

    # Load the first 5 test images for evaluation
    test_data = get_dataset(
        batch_size=1,
        img_size=(128, 128),
        input_img_paths=(args.input_img_paths/'test/images').glob('*.png'),
        target_img_paths=(args.target_img_paths/'test/masks').glob('*_mask.png'),
    )
    # test_image_paths = sorted(list((args.input_img_paths / 'test/images').glob('*.png')))[:5]
    # test_mask_paths = sorted(list((args.target_img_paths / 'test/masks').glob('*_mask.png')))[:5]
    
    output_dir = Path(args.model_dir) / args.model_name
    output_dir.mkdir(exist_ok=True, parents=True)


    # Initialize the custom loss function with class weights
    weight_0 = 0.5238804478064116
    weight_1 = 85.2202101955135
    class_weights = {0: weight_0, 1: weight_1}

    # Define the custom loss function
    loss_fn = weighted_binary_crossentropy(class_weights)

    # Define the model
    x, y = next(iter(train_data))
    model = UNet(in_channels=x.shape[-1],
                 num_filters_per_layer=16,
                 activation=args.activation,
                 dropout=args.dropout,
                 out_channels=y.shape[-1])
    
    # Custom metric for IoU, since default was producing one value for all epochs
    class CustomMetricsCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_data):
            super().__init__()
            self.val_data = val_data

        def on_epoch_end(self, epoch, logs=None):
            y_true = np.concatenate([y.numpy() for x, y in self.val_data], axis=0).flatten()
            y_pred = self.model.predict(self.val_data).flatten()
            iou = compute_iou(y_true, y_pred)
            logs['val_iou'] = iou.numpy()

    # Compile the model with custom metrics and loss function
    model.compile(
        optimizer=tf.keras.optimizers.get(
            {
                "class_name": args.optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        ),
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.MeanIoU(num_classes=2),
            # tf.keras.metrics.BinaryCrossentropy(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            # tf.keras.metrics.F1Score(),
            compute_iou,
        ]
    )
            
    # Custom callback to apply threshold and compute metrics
    class ThresholdCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            val_predictions = self.model.predict(val_data)
            val_predictions_thresholded = (val_predictions > 0.5).astype(np.float32)

            # Flatten the predictions and labels for metric computation
            y_true = np.concatenate([y.numpy() for x, y in val_data], axis=0).flatten()
            y_pred = val_predictions_thresholded.flatten()
            
            # Compute metrics
            accuracy = tf.keras.metrics.BinaryAccuracy()
            precision = tf.keras.metrics.Precision()
            recall = tf.keras.metrics.Recall()

            accuracy.update_state(y_true, y_pred)
            precision.update_state(y_true, y_pred)
            recall.update_state(y_true, y_pred)

            print(f'Epoch {epoch+1}: val_accuracy: {accuracy.result().numpy()},'
                  f'val_precision: {precision.result().numpy()},'
                  f'val_recall: {recall.result().numpy()},'
                  )
                  

            # Reset metrics
            accuracy.reset_state()
            precision.reset_state()
            recall.reset_state()

            # Print confusion matrix for a sample
            if epoch == args.max_epochs - 1:
                y_true_sample = y_true
                y_pred_sample = y_pred
                cm = confusion_matrix(y_true_sample, y_pred_sample)
                plt.figure(figsize=(10, 7))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                confusion_matrix_path = Path(args.model_dir) / args.model_name / "visualized_results" / "confusion_matrix.png"
                confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(confusion_matrix_path)
                plt.show()

    # Create the output directory
    output_dir = Path(args.model_dir) / args.model_name
    output_dir.mkdir(exist_ok=True, parents=True)  

    with open(output_dir / "args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=args.patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ThresholdCallback(),
            CustomMetricsCallback(val_data),
        ],
    )
    
    # Calling the function to visualize the model results
    visualize_results(model, test_data, output_dir)
    
    # Save the model
    model.save(output_dir / f"{args.model_name}.keras")

    # Save the training history with a plot to go along with it
    pd.DataFrame(history.history).to_csv(output_dir / "history.csv")
    print(history.history)
    plot_history(history, output_dir / "history.png")
