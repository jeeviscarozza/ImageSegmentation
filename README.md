# ImageSegmentation
For images of cracks in infrastructure

This repository contains my implementation of a U-Net model for the purpose of semantic image segmentation. The training script allows for configurations using command-line arguments. Additionally, there are functions to visualize the training results and predictions.

Dataset
The dataset used to train the model is available at Dataverse Harvard https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGIEBY. It consists of 10 sub-datasets, each preprocessed and resized to 400x400 pixels. For my project, I have further resized all images to 128x128.

The first script that should be run is the data.py, to split the data into appropriate folders.

For some data exploration, the preprocessdata notebook can be ran. Or, the script, visualize.sh can run the file, visualize_input.py, using the command bash visualize.sh for a quick way to visualize the training data. To run with your data, all that is required is to change the path to the input and target images to the folder that contains your data within the repository.

Training Process
The training process involved running multiple experiments with different hyperparameters to identify the best configuration for the U-Net model. Steps can be seen and re-ran using the training_commands.sh file.

Default Parameters
For default parameters and more details, please refer to the utils.py file in the src code. 

Evaluation
Model is evaluated directly using the train.py file. Ideally however, the model should be able to be evaluated using the eval_commands.sh script. See eval.py for details. Here is an example command.
python src/cli.py \
    --eval \
    --model_names unet_1 \
    --image_index 0 \
    --model_dir models 

All results are saved directly to the models folder, within the folder for each respective model.
