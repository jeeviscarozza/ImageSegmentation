#!/bin/bash

# Train a unet networks, each with a different hyperparameters and compare their performance
# this first one is for testing to ensure training works, and images are saved correctly
python src/cli.py --train --model_name=unet_test --input_img_paths=preprocessed_data --target_img_paths=preprocessed_data --loss_fn=loss_fn --optimizer=adam --max_epochs=1 --batch_size=64 --learning_rate=0.001 --activation=relu --model_dir=models

# now testing two different learning rates (patience is set to 5 as a default)
python src/cli.py --train --model_name=unet_1 --input_img_paths=preprocessed_data --target_img_paths=preprocessed_data --loss_fn=loss_fn --optimizer=adam --max_epochs=100 --batch_size=64 --learning_rate=0.001 --activation=relu --model_dir=models
python src/cli.py --train --model_name=unet_1_lr --input_img_paths=preprocessed_data --target_img_paths=preprocessed_data --loss_fn=loss_fn --optimizer=adam --max_epochs=100 --batch_size=64 --learning_rate=0.0001 --activation=relu --model_dir=models

# now with the selected learning rate and a larger batch size
python src/cli.py --train --model_name=unet_batch_128 --input_img_paths=preprocessed_data --target_img_paths=preprocessed_data --loss_fn=loss_fn --optimizer=adam --max_epochs=100 --batch_size=128 --learning_rate=0.001 --activation=relu --model_dir=models

# now with the selected learning rate and a smaller batch size
python src/cli.py --train --model_name=unet_batch_32 --input_img_paths=preprocessed_data --target_img_paths=preprocessed_data --loss_fn=loss_fn --optimizer=adam --max_epochs=100 --batch_size=32 --learning_rate=0.001 --activation=relu --model_dir=models

# now comparing different dropout values
python src/cli.py --train --model_name=unet_dropout_0.1 --input_img_paths=preprocessed_data --target_img_paths=preprocessed_data --loss_fn=loss_fn --optimizer=adam --max_epochs=100 --batch_size=64 --learning_rate=0.001 --dropout=0.1 --activation=relu --model_dir=models
python src/cli.py --train --model_name=unet_dropout_0.3 --input_img_paths=preprocessed_data --target_img_paths=preprocessed_data --loss_fn=loss_fn --optimizer=adam --max_epochs=100 --batch_size=64 --learning_rate=0.001 --dropout=0.3 --activation=relu --model_dir=models
python src/cli.py --train --model_name=unet_dropout_0.5 --input_img_paths=preprocessed_data --target_img_paths=preprocessed_data --loss_fn=loss_fn --optimizer=adam --max_epochs=100 --batch_size=64 --learning_rate=0.001 --dropout=0.5 --activation=relu --model_dir=models

# with the regular binary crossentropy loss
#code was adjusted to allow for the default binary crossentropy loss
python src/cli.py --train --model_name=unet_binary_crossentropy --input_img_paths=preprocessed_data --target_img_paths=preprocessed_data --loss_fn=binary_crossentropy --optimizer=adam --max_epochs=100 --batch_size=64 --learning_rate=0.001 --activation=relu --model_dir=models

# with a batch size of 32, the model had the least amount of overfitting
# training again with this batch size and with the best dropout value of 0.1 to see if I can produce better results
python src/cli.py --train --model_name=unet_better --input_img_paths=preprocessed_data --target_img_paths=preprocessed_data --loss_fn=loss_fn --optimizer=adam --max_epochs=100 --batch_size=32 --learning_rate=0.001 --dropout=0.1 --activation=relu --model_dir=models
