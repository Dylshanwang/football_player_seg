from dataset import data_split, Custom_Dataset
import segmentation_models_pytorch as smp
from config import CFG, seed_everything
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import albumentations as A
from segmentation_models_pytorch.encoders import get_preprocessing_fn


def get_transforms():
    return A.Compose(
        [
            A.HueSaturationValue(
                p=1.0,
                hue_shift_limit=(-20, 20),
                sat_shift_limit=(-30, 30),
                val_shift_limit=(-20, 20),
            ),
            A.HorizontalFlip(p=0.5),
        ],
        p=1.0,
    )


def create_dataloaders() -> list:
    indexes = list(range(CFG.N_IMAGES))
    images, masks = data_split()

    train_dataset = Custom_Dataset(
        indexes=indexes[: int(CFG.N_IMAGES * CFG.train_size)],
        images=images,
        masks=masks,
        transform=get_transforms(),
        preprocess=get_preprocessing_fn("resnet34", pretrained="imagenet"),
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=CFG.batch_size, shuffle=True
    )

    test_dataset = Custom_Dataset(
        indexes=indexes[int(CFG.N_IMAGES * CFG.train_size) :],
        images=images,
        masks=masks,
        preprocess=get_preprocessing_fn("resnet34", pretrained="imagenet"),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def train():
    seed_everything(CFG.seed)
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1
    )
    train_dataloader, test_dataloader = create_dataloaders()
    model.to(CFG.device)

    loss_fn = BCEWithLogitsLoss()
    optimiser = Adam(model.parameters(), lr=CFG.lr)

    train_loss_history, val_loss_history = [], []

    for epoch in range(CFG.n_epochs):
        train_loss = 0
        model.train()

        for batch in train_dataloader:
            inputs = batch["image"].to(CFG.device)
            labels = batch["mask"].to(CFG.device)

            optimiser.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            train_loss += loss.item()

            loss.backward()
            optimiser.step()

        train_loss /= len(train_dataloader)
        train_loss_history.append(train_loss)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = batch["image"].to(CFG.device)
                labels = batch["mask"].to(CFG.device)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(test_dataloader)
        val_loss_history.append(val_loss)

        print(
            "Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}".format(
                epoch + 1, CFG.n_epochs, train_loss, val_loss
            )
        )

    torch.save(model.state_dict(), "latest_model.pt")
