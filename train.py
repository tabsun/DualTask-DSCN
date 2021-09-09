import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from reader import get_dataloader
from model import CDNet34
from engine import Trainer, Tester
from cdloss import cdloss

def test(batch_size, save_path):
    val_loader   = get_dataloader(batch_size, mode='val', shuffle=False)
    model=CDNet34(in_channels=3, out_channels=1)

    tester = Tester(model, save_path)
    tester.test(val_loader)

def train(batch_size, save_path):
    train_loader = get_dataloader(batch_size, mode='train', shuffle=True)
    val_loader   = get_dataloader(8, mode='val', shuffle=False)
    model=CDNet34(in_channels=3, out_channels=1)
    optimizer = optim.SGD(params=model.parameters(), lr=1e-2, momentum=0.9,weight_decay=1e-4)
    #optimizer=optim.Adam(params=model.parameters(),lr=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], 0.1)
    trainer = Trainer(model, optimizer, cdloss ,save_freq=1, save_dir=save_path)
    trainer.loop(100, train_loader, val_loader, scheduler=scheduler)

if __name__ == '__main__':
    BATCH_SIZE = 4
    #SAVE_PATH = './results'
    #train(BATCH_SIZE, SAVE_PATH)

    MODEL_PATH = './results/model_72.pkl'
    test(BATCH_SIZE, MODEL_PATH)
