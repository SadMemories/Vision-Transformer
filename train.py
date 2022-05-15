import os
import sys
import json
import math
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from my_dataset import split_train_valid, MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from vit import vit_base_patch16_224
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model, optimizer, train_dataloader, device, epoch):

    correct_num = 0
    total_num = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    model.train()
    data_loader = tqdm(train_dataloader, file=sys.stdout)
    for step, (images, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)

        pred_ind = torch.max(pred, dim=1)[1]
        loss = criterion(pred, labels)

        iter_corr_num = torch.eq(pred_ind, labels).data.sum()
        total_num += images.shape[0]
        correct_num += iter_corr_num

        loss.backward()
        total_loss += loss.item()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               total_loss / (step + 1),
                                                                               correct_num / total_num)

        optimizer.step()
    return total_loss / (step + 1), correct_num / total_num


@torch.no_grad()
def valid_one_epoch(model, valid_dataloader, device, epoch):
    correct_num = 0
    total_num = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    model.eval()
    data_loader = tqdm(valid_dataloader, file=sys.stdout)
    for step, (images, labels) in enumerate(data_loader):

        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)

        pred_ind = torch.max(pred, dim=1)[1]
        loss = criterion(pred, labels)

        iter_corr_num = torch.eq(pred_ind, labels).data.sum()
        total_num += images.shape[0]
        correct_num += iter_corr_num

        total_loss += loss.item()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               total_loss / (step + 1),
                                                                               correct_num / total_num)

    return total_loss / (step + 1), correct_num / total_num


def train(args):
    tb_writer = SummaryWriter()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_img_path, train_img_label, valid_img_path, valid_img_label = split_train_valid(args.data_path)

    data_transform = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_dataset = MyDataset(img_path_list=train_img_path,
                              img_label_list=train_img_label, transform=data_transform['train'])
    valid_dataset = MyDataset(img_path_list=valid_img_path,
                              img_label_list=valid_img_label, transform=data_transform['valid'])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, collate_fn=MyDataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, collate_fn=MyDataset.collate_fn)

    model = vit_base_patch16_224(args.num_classes)
    # optimizer = optim.SGD(model)
    if args.weight:
        assert os.path.exists(args.weight), f"{args.weight} path is not exists..."
        model_weight = torch.load(args.weight, map_location=device)

        del_weight = ['head.weight', 'head.bias']

        for del_key in del_weight:
            del model_weight[del_key]
        model.load_state_dict(model_weight, strict=False)

    if args.freeze_layer:
        for name, para in model.named_parameters():
            if 'head' not in name:
                para.requires_grad(False)

    model.to(device)
    train_para = [para for para in model.parameters() if para.requires_grad]
    optimizer = optim.SGD(train_para, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    cos_lr = lambda x: args.lrf + (1 + math.cos(math.pi * x / args.epochs)) \
                       * ((1 - args.lrf) / 2)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cos_lr)

    for epoch in range(args.epochs):

        train_loss, train_acc = train_one_epoch(model, optimizer, train_dataloader, device, epoch)
        scheduler.step()

        valid_loss, valid_acc = valid_one_epoch(model, valid_dataloader, device, epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], valid_loss, epoch)
        tb_writer.add_scalar(tags[3], valid_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weight/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/disk/wangweijia/data/FlowData/flower_photos")
    parser.add_argument('--device', type=str, default="cuda:2")

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--weight', type=str, default='./weight/vit_base_patch16_224.pth')
    parser.add_argument('--freeze-layer', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', default=0.01, type=float)
    parser.add_argument('--epochs', type=int, default=10)
    opt = parser.parse_args()
    train(opt)