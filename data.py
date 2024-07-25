import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from keras import layers
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory as tf_data
from tensorflow import data as tf_data
import os
import random
from shutil import copyfile
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
from utils import get_training_args
import cv2

def split_data(images_dir, masks_dir, parent_dir, train_size=0.8, val_size=0.10, test_size=0.1, random_state=42):
    # Create a new directory for preprocessed data within the same parent directory
    preprocessed_dir = os.path.join(parent_dir, "preprocessed_data")
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Create directories for train, val, and test sets within the preprocessed data folder
    train_dir = os.path.join(preprocessed_dir, "train")
    val_dir = os.path.join(preprocessed_dir, "val")
    test_dir = os.path.join(preprocessed_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of image file paths
    image_files = os.listdir(images_dir)
    # Shuffle the list for randomness
    random.seed(random_state)
    random.shuffle(image_files)  

    # Split dataset into train, val, and test sets
    train_files, test_val_files = train_test_split(image_files, test_size=val_size+test_size, random_state=random_state)
    val_files, test_files = train_test_split(test_val_files, test_size=test_size/(val_size+test_size), random_state=random_state)

    # Define function to copy files to respective directories
    def copy_files(file_list, source_dir, dest_dir):
        for file in file_list:
            image_path = os.path.join(source_dir, file)
            mask_path = os.path.join(masks_dir, file)  # Assuming mask filenames match image filenames
            dest_image_path = os.path.join(dest_dir, file)
            dest_mask_path = os.path.join(dest_dir, f"{os.path.splitext(file)[0]}_mask.png")
            copyfile(image_path, dest_image_path)
            copyfile(mask_path, dest_mask_path)

    # Copy files to train, val, and test directories within the preprocessed data folder
    copy_files(train_files, images_dir, train_dir)
    copy_files(val_files, images_dir, val_dir)
    copy_files(test_files, images_dir, test_dir)

    # Display how many images are in each set
    print(f"Train set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    print(f"Test set: {len(test_files)} images")

    return train_dir, val_dir, test_dir

def get_dataset(
    batch_size,
    img_size,
    input_img_paths,
    target_img_paths,
    max_dataset_len=None,
):
    """Function that returns a TF Dataset. This loads images and masks, resizing, and batching."""

    def load_image(image_path):
        image_path = tf.convert_to_tensor(image_path, dtype=tf.string)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to [0,1] range
        image = tf.image.resize(image, img_size) 
        return image

    def load_mask(mask_path):
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=1)
        mask.set_shape([None, None, 1])
        mask = tf.image.convert_image_dtype(mask, tf.float32)  # Convert to [0,1] range
        mask = tf.image.resize(mask, img_size)
        mask = tf.where(mask > 0, 1.0, 0.0)
        return mask

    def load_image_masks(input_image_path, target_image_path):
        input_image = load_image(input_image_path)
        target_image = load_mask(target_image_path)
        input_image = tf.image.resize(input_image, img_size)
        target_image = tf.image.resize(target_image, img_size)
        return input_image, target_image
    
    input_img_paths = [str(p) for p in input_img_paths]
    target_img_paths = [f"{p.split('.')[0].replace('images', 'masks')}_mask.{p.split('.')[1]}" for p in input_img_paths]

    # TODO: from_tensor_slices may not do what we want here
    dataset = tf.data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(load_image_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)

def image_masks(data_dir):
    data_dir = Path(data_dir) / "masks" 
    images = []
    masks = []

    for filename in data_dir.glob("*.png"):
        
        # Check if the current file is a mask
        if "_mask" in filename.stem:
            continue 

        img = cv2.imread(str(filename))
        images.append(img)

        mask_filename = filename.stem + "_mask.png"
        mask_path = filename.parent / mask_filename
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
            
    print(f"Number of images: {len(images)}")  # Printing the number of images stored
    print(f"Number of masks: {len(masks)}")  # Printing the number of masks stored
    return np.array(images), np.expand_dims(np.array(masks), axis=-1) 

