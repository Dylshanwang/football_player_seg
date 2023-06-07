# Football Player Segmentation
Mini project on the football player segmentation dataset

## Install dependencies

Run `pip install -r requirements.txt` in the root directory of the project

## The Data

Download from: https://www.kaggle.com/datasets/ihelon/football-player-segmentation and extract the zip to the root of the project

## Training The Model

Run `python main.py` in the terminal and a file called `latest_model.pt` will be saved to the root directory

## Running Inference

Place image to run inference on in the root directory of the project. An image named `seg_image.png` will be saved to the root directory

For example: `python main.py latest_model.pt test_image.jpg`

Play around with the hyperparameters and save different model files to run inference with.