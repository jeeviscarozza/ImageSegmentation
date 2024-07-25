import tensorflow as tf
import matplotlib.pyplot as plt
from data import get_dataset
from utils import get_training_args

args = get_training_args()
# Load the dataset
train_data = get_dataset(
    batch_size=args.batch_size,
    img_size=(128, 128),
    input_img_paths=(args.input_img_paths/'train/images').glob('*.png'),
    target_img_paths=(args.target_img_paths/'train/masks').glob('*_mask.png'),
)

# Function to visualize input and target images
def visualize_batch(dataset):
    for input_image, target_image in dataset.take(1):  # Taking one batch
        input_image = input_image.numpy()
        target_image = target_image.numpy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(input_image[0])  # Display the first image in the batch
        axes[0].set_title('Input Image')
        axes[1].imshow(target_image[0, :, :, 0], cmap='gray')  
        axes[1].set_title('Target Mask')
        plt.show()
        plt.close()

visualize_batch(train_data)
