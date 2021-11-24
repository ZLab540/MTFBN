# coding=utf-8
import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from utils import *
from train_data import TrainData
from val_data import ValData
from model import ours
from loss import *

# Training settings
parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--tag", type=str, help="tag for this training")
parser.add_argument("--train", default="./data/train/indoor/", type=str,
                    help="path to load train datasets(default: none)")
parser.add_argument("--test", default="./data/test/SOTS/indoor/", type=str,
                    help="path to load test datasets(default: none)")
parser.add_argument("--batchSize", type=int, default=8, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=100, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0002, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100, help="step to test the model performance. Default=2000")
parser.add_argument("--cuda", action="store_true",default=1,help="Use cuda?")
parser.add_argument("--gpus", type=int, default=1, help="nums of gpu to use")
parser.add_argument("--resume", default=" ", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 4")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def main():
    global opt, logger, model, criterion,ms_ssim_loss,best_psnr
    opt = parser.parse_args()
    print(opt)
    import random
    opt.best_psnr = 0
    logger = SummaryWriter("runs/")
    
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    opt.seed = 7561
    print("Random Seed: ", opt.seed)
    opt.seed_python = 7455
    random.seed(opt.seed_python)
    print("Random Seed_python: ", opt.seed_python)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print("==========> Loading datasets")    
    train_data_dir = opt.train
    val_data_dir = opt.test
    # --- Load training data and validation/test data --- #
    training_data_loader = DataLoader(TrainData([256, 256], train_data_dir), batch_size=opt.batchSize, shuffle=True, num_workers=12,drop_last=True)
    indoor_test_loader = DataLoader(ValData(val_data_dir), batch_size=1, shuffle=False, num_workers=12)

    print("==========> Building model")
    model = ours()
    criterion_mse = nn.MSELoss(size_average=True)
    criterion_L1 = nn.L1Loss(size_average=True)
    ms_ssim_loss = MS_SSIM()
    
    # --- Set the GPU --- #
    print("==========> Setting GPU")
    if cuda:
        model = nn.DataParallel(model, device_ids=[i for i in range(opt.gpus)]).cuda()
        criterion_mse = criterion_mse.cuda()
        criterion = criterion_L1.cuda()
        ms_ssim_loss = ms_ssim_loss.cuda()
    print("==========> Setting Optimizer")
    # --- Build optimizer --- #
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    print("==========> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
        train(training_data_loader, indoor_test_loader,optimizer, epoch)
        test(indoor_test_loader, epoch)

def train(training_data_loader, indoor_test_loader,optimizer, epoch):
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    for iteration, batch in enumerate(training_data_loader, 1):
        num_step = len(training_data_loader) * opt.nEpochs
        steps = len(training_data_loader) * (epoch-1) + iteration
        optimizer.param_groups[0]['lr'] = 0.0002 * (1 - float(steps) / num_step) ** 0.9
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        steps = len(training_data_loader) * (epoch-1) + iteration
        data,label = \
            Variable(batch[0]), \
            Variable(batch[1], requires_grad=False)
        if opt.cuda:
            data = data.cuda()
            label = label.cuda()
        else:
            data = data.cpu()
            label = label.cpu()
        x_fusion = model(data)
        ssim_loss = 3 - ms_ssim_loss(x_fusion[0], label) - ms_ssim_loss(x_fusion[1], label) - ms_ssim_loss(x_fusion[2], label)
        loss_x_fusion = criterion(x_fusion[0], label)+criterion(x_fusion[1], label)+criterion(x_fusion[2], label)
        loss = 0.65*loss_x_fusion + 0.35*ssim_loss
        ssim_loss = 3 - ms_ssim_loss(x_fusion[3], label) - ms_ssim_loss(x_fusion[4], label) - ms_ssim_loss(x_fusion[5], label)
        loss_x_fusion = criterion(x_fusion[3], label)+criterion(x_fusion[4], label)+criterion(x_fusion[5], label)
        loss_fb = 0.65*loss_x_fusion + 0.35*ssim_loss
        loss_final = loss_fb + 0.1*loss
        loss_final.backward()
        optimizer.step()

def test(test_data_loader, epoch):
    psnrs = []
    ssims = []
    psnrs_add = []
    ssims_add = []
    psnrs_dir = []
    ssims_dir = []
    for iteration, batch in enumerate(test_data_loader, 1):
        model.eval()
        with torch.no_grad():
            data, label = \
            Variable(batch[0]), \
            Variable(batch[1])

        if opt.cuda:
            data = data.cuda()
            label = label.cuda()
        else:
            data = data.cpu()
            label = label.cpu()

        with torch.no_grad():
            x_fusion = model(data)
        output = torch.clamp(x_fusion[5], 0., 1.)
        # --- Calculate the average PSNR --- #
        psnrs.extend(to_psnr(output, label))
        # --- Calculate the average SSIM --- #
        ssims.extend(to_ssim_skimage(output, label))
    psnr_mean = sum(psnrs) / len(psnrs)
    ssim_mean = sum(ssims) / len(ssims)
    if opt.best_psnr < psnr_mean:
        opt.best_psnr = psnr_mean
        save_checkpoint(model, epoch, 'best_model')
    print("test  epoch %d psnr: %f ssim: %f " % (epoch, psnr_mean,ssim_mean))
    print("pyotrch_seed %d python_seed %d best_psnr %f" % (opt.seed, opt.seed_python,opt.best_psnr))

if __name__ == "__main__":
    os.system('clear')
    main()
