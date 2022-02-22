import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import BasicCNN
from utils import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from loss import *
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = 8
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data,target) in enumerate(loop):
        data = data.to(device=device)
        target = target.float().to(device=device)

    #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)

            loss = loss_fn(predictions,target)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    model = BasicCNN(input_channels=1,output_channels=10)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = FocalLoss(gamma=0,logits=True)
    #loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    
    train_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH),
            ToTensorV2(),
        ]
    )
    train_loader, val_loader = get_loaders(
        train_transform=train_transform,
        val_transform=val_transform)

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        checkpoint = {
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            }
        check_accuracy(val_loader,model,epoch,batch_size=16,device=device)
if __name__ =="__main__":
    main()