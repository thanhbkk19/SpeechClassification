import enum
import cv2 as cv
import torch
import numpy as np
import pandas as pd
from PIL import Image
from dataset import MNIST_data
from torch.utils.data import Dataset, DataLoader
import torchvision
import cv2


def save_checkpoints(state, file_name="checkpoint.pth.tar"):
    print("=========> saving checkpoint")
    torch.save(state,file_name)

def load_checkpoints(checkpoint, model):
    print("==========> loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    return model

def get_loaders(
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    batch_size=16
):
    train_ds = MNIST_data(data_path="/home/gumiho/project/SpeechClassification/free-spoken-digit-dataset/recordings",transform=train_transform,type="train")
    val_ds = MNIST_data(data_path="/home/gumiho/project/SpeechClassification/free-spoken-digit-dataset/recordings",transform=val_transform, type = "val")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader,val_loader

def check_accuracy(loader, model, epoch, batch_size = 8, device="cuda"):
    num_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            for i in range(len(y)):
                if np.argmax(preds[i]) == np.argmax(y[i]):
                    num_correct+=1
                total+=1
    print(f"EPOCH: {epoch} Got {num_correct}/{total} ----> accuracy = {num_correct/total:.2f}")
    model.train()


def val_loss(loader, model, loss_fn, epoch, device="cuda"):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            y = y.float().to(device = device)
            y = y.permute(0,3,1,2)
            preds = model(x)
            loss = loss_fn(y, preds)
            loss_total+= loss
    print(f"total loss in val dataset at epoch {epoch}: {loss_total}")
    return loss_total

