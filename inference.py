import cv2
import numpy as np
from model import get_transforms
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
import segmentation_models_pytorch as smp
from config import CFG
import matplotlib.pyplot as plt


def infer(model_name: str, image_name: str):
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1
    )
    model.load_state_dict(torch.load(model_name))
    model.to(CFG.device)

    transform = get_transforms()
    preprocess = get_preprocessing_fn("resnet34", pretrained="imagenet")

    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    image = transform(image=image)["image"]
    image = preprocess(image)
    image = torch.tensor(image, dtype=torch.float64)
    image = image.permute(2, 0, 1)
    image = image.float()

    with torch.no_grad():
        image = image.to(CFG.device)
        pred = model(image.unsqueeze(0))
        np_pred = pred.detach().cpu().numpy()

    np_pred = (np_pred > 0.5).astype(np.uint8)

    plt.imsave("seg_image.png", np_pred[0][0], format="png", cmap="bwr")
