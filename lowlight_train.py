import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    start_time = time.time()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    DCE_net = model.enhance_net_nopool().cuda()

    DCE_net.apply(weights_init)
    if config.load_pretrain == True:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    writer = SummaryWriter()

    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()

    L_exp = Myloss.L_exp(16, 0.55)
    L_TV = Myloss.L_TV()

    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    DCE_net.train()

    for epoch in range(config.num_epochs):
        print("EPOCH" , epoch+1)
        for iteration, img_lowlight in enumerate(train_loader):

            img_lowlight = img_lowlight.cuda()

            enhanced_image, A = DCE_net(img_lowlight)

            Loss_TV = 1000 * L_TV(A)

            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))

            loss_col = 5 * torch.mean(L_color(enhanced_image))

            loss_exp = 10 * torch.mean(L_exp(enhanced_image))

            #loss_ssim = 1 * (1-L_ssim(img_lowlight, enhanced_image))

            #loss_laplacian = 1 * L_laplacian(img_lowlight,enhanced_image)

            # best_loss
            loss = Loss_TV + loss_spa + loss_col + loss_exp
            #

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

        writer.add_scalar('Loss/Train', loss.item(), global_step=epoch)

        for name, param in DCE_net.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'{name}.grad', param.grad, global_step=epoch)

        with torch.no_grad():
            for name, param in DCE_net.named_parameters():
                if param.requires_grad:
                    variance = param.var().item()
                    writer.add_scalar(f'Variance/{name}', variance, global_step=epoch)

            writer.add_scalar('IntermediateOutput/Variance', variance, global_step=epoch)

    writer.close()

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"total time：{elapsed_time} seconds")


if __name__ == "__main__":
    writer = SummaryWriter(log_dir="runs/data1", flush_secs=120)
    for n_iter in range(100):
        writer.add_scalar(tag='Loss/train',
                          scalar_value=np.random.random(),
                          global_step=n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.close()
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="snapshots/Epoch299.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
