import random
import os
import numpy as np
import torch


class CFG:
    seed = 42
    train_size = 0.8
    batch_size = 4
    lr = 0.001
    n_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_IMAGES = 512
    IMAGE_SIZE = 512
    INPUT_IMAGE_SIZE = (1920, 1080)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
