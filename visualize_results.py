import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from data import get_dataset
import numpy as np
from utils import get_training_args
from models import UNet
from train import weighted_binary_crossentropy, compute_iou

args = get_training_args()

def evaluate_single_image(args, model_path, input_image_path, target_mask_path):
    # Load your trained model
    #model = tf.keras.models.load_model(
    #filepath="/Users/jeeviscarozza/Documents/Spring2024Physics/AICourse/FinalProject/models/unet_dropout_0.3/unet_dropout_0.3.keras")
    custom_objects = {
        'UNet': UNet,
        'loss_fn': weighted_binary_crossentropy,
        'compute_iou': compute_iou,
        #'BatchNormalization': tf.keras.layers.BatchNormalization,
        #'Conv2DTranspose': tf.keras.layers.Conv2DTranspose
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    visualize_results(model, input_image_path, target_mask_path)

def visualize_results(model, input_image_path, target_mask_path, img_size=(128, 128)):
    input_image = cv2.imread(str(input_image_path))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, img_size)
    input_image = input_image / 255.0  # Normalize to [0, 1]

    target_mask = cv2.imread(str(target_mask_path), cv2.IMREAD_GRAYSCALE)
    target_mask = cv2.resize(target_mask, img_size)
    target_mask = target_mask / 255.0  # Normalize to [0, 1]

    input_image = input_image[None, ...]  # Add batch dimension

    predicted_mask = model.predict(input_image)
    predicted_mask = tf.round(predicted_mask[0])  # Remove batch dimension and round the prediction

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(tf.squeeze(input_image))
    plt.title("Input Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(tf.squeeze(target_mask), cmap='gray')
    plt.title("Target Mask")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(tf.squeeze(predicted_mask), cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")
    
    plt.show()

if __name__ == "__main__":

    model_name = "unet_default_lr"  
    input_image_path = "/Users/jeeviscarozza/Documents/Spring2024Physics/AICourse/FinalProject/preprocessed_data/test/images/CRACK500_20160222_163930_1921_1081.png" 
    target_mask_path = "/Users/jeeviscarozza/Documents/Spring2024Physics/AICourse/FinalProject/preprocessed_data/test/masks/CRACK500_20160222_163930_1921_1081_mask.png" 

    model_path = f"{args.model_dir}/{model_name}/{model_name}.keras"
    visualize_results(model_path, input_image_path, target_mask_path)
    evaluate_single_image(args,model_path, input_image_path, target_mask_path)

