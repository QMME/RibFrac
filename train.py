from functools import partial

import os
import torch
from torch._C import device
import torch.nn as nn

from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset import RibFracDataset
import transforms as trans
from metrics import dice, recall, precision, fbeta_score
from model import UNet
#from model.unet import UNet
from loss import MixLoss, DiceLoss
from util import progress_bar

from torch.optim.lr_scheduler import CosineAnnealingLR

def train(dataloader, scheduler, criterion, model, device, scaler):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        scheduler.optimizer.zero_grad()

        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(scheduler.optimizer)
        scaler.update()
        progress_bar(batch_idx, len(dataloader), 'Loss: %.3f '
            % loss)

def val(dataloader, criterion, model, device, dir_path):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            targets = targets.to(device)
            outputs = model(inputs.to(device))
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)
            progress_bar(batch_idx, len(dataloader))

    dice = criterion[0](y_true,y_score)
    recall = criterion[1](y_true,y_score)
    precision = criterion[2](y_true,y_score)
    fbeta_score = criterion[3](y_true,y_score)

    
    path = os.path.join(dir_path, 'model_weights.pth')
    print('Saved state: dice {}, recall {}, precision {}, fbeta_score {}'.format(dice, recall, precision, fbeta_score))
    torch.save(model.state_dict(), path)


def main(args):
    train_image_dir = args.train_image_dir
    train_label_dir = args.train_label_dir
    val_image_dir = args.val_image_dir
    val_label_dir = args.val_label_dir

    model = UNet(in_channels = 1, num_classes = 1)
    model = model.cuda()

    if(args.resume_dir):
        restore_model_path = os.path.join(args.resume_dir)
        resumed = torch.load(restore_model_path)
        model.load_state_dict(resumed)
        print('==> Resuming model')

    batch_size = 2
    num_workers = 4

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    lr_scheduler = CosineAnnealingLR(optimizer, 100, 0)
    scaler = torch.cuda.amp.GradScaler()
    criterion = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)

    thresh = 0.1
    recall_partial = partial(recall, thresh=thresh)
    precision_partial = partial(precision, thresh=thresh)
    fbeta_score_partial = partial(fbeta_score, thresh=thresh)

    metrics=[dice, recall_partial, precision_partial, fbeta_score_partial]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transforms_train = [
        trans.Window(-200, 1000),
        trans.MinMaxNorm(-200, 1000),
        trans.Noise(0, 0.01),
        trans.Flip()
    ]

    transforms_test = [
        trans.Window(-200, 1000),
        trans.MinMaxNorm(-200, 1000)
    ]
    dataset_train = RibFracDataset(train_image_dir, train_label_dir,
        transforms=transforms_train)
    dataloader_train = DataLoader(dataset_train, batch_size, False,
            num_workers=num_workers, collate_fn = dataset_train.collate_fn)
    dataset_val = RibFracDataset(val_image_dir, val_label_dir,
        transforms=transforms_test)
    dataloader_val = DataLoader(dataset_val, batch_size, False,
            num_workers=num_workers, collate_fn = dataset_val.collate_fn)
    
    for epoch in range(args.epoch):
        print("epoch {}".format(epoch))
        train(dataloader_train, lr_scheduler, criterion, model, device, scaler)
        lr_scheduler.step()
        val(dataloader_val, metrics, model, device, args.save_dir)

if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", default='data/seq-train/',
        help="The training image nii directory.")
    parser.add_argument("--train_label_dir", default='data/seq-label/',
        help="The training label nii directory.")
    parser.add_argument("--val_image_dir", default='data/seq-train/',
        help="The validation image nii directory.")
    parser.add_argument("--val_label_dir", default='data/seq-label/',
        help="The validation label nii directory.")
    parser.add_argument("--save_dir", default='./',
        help="Where to save the trained model.")
    parser.add_argument("--epoch", type=int, default=200,
        help="epochs to train the model")
    parser.add_argument("--resume_dir", default=None,
        help="epochs to train the model")
    args = parser.parse_args()

    main(args)