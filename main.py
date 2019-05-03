################################   NOTICE   ###########################################

# This code is based from Mean teachers are better role models
#https://github.com/CuriousAI/mean-teacher

###################################################################################

import argparse
import os
import pickle
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.datasets as datasets
from data.preprocessing import *
from data.data_loader import *
from Model.model import *
from Model.utils import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# input setting
parser.add_argument('--dataset', type=str, default='../pytorch/data-local')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--labels', default='00.txt', type=str, help='list of image labels (default: based on directory structure)')
parser.add_argument('--cached_data_file', default='cifar.p', help='Cached file name')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu_id', type=str, default='0')
# train setting
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--num_epochs', default=600, type=int, help='number of total epochs to run')
parser.add_argument('--labeled_batchsize', default=20, type=int, help='mini-batch size ')
parser.add_argument('--unlabeled_batchsize', default=80, type=int, help="labeled examples per minibatch (default: no constrain)")
parser.add_argument('--test_batchsize', default=100, type=int, help="labeled examples per minibatch (default: no constrain)")

parser.add_argument('--lr', default=0.04, type=float, help='initial learning rate')
parser.add_argument('--lamdaC', default=1.0, type=float, help='classification weight')
parser.add_argument('--lamdaR', default=0.25, type=float, help='reconstruction weight')
parser.add_argument('--lamdaL', default=0.5, type=float, help='inter layer reconstruction weight')
parser.add_argument('--lamdaS', default=1, type=float, help='stability weight')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='use nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

############# Etc setting ######################################
parser.add_argument('--output', type=str, default='result')
parser.add_argument('--name', type=str, default='cifar10shackRes')

args = parser.parse_args()
save_dir = os.path.join(args.output, args.name)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# output setting
use_cuda = torch.cuda.is_available()
if use_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if __name__ == '__main__':
    #########################setting cifar 10 semi sp dataset ###########################################
    if not os.path.isfile(args.cached_data_file):
        img_dir = os.path.join(args.dataset, 'images/cifar/cifar10/by-image/train+val')
        label_dir = os.path.join(args.dataset, 'labels/cifar10/1000_balanced_labels', args.labels)
        with open(label_dir) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())

        labeled_sets, unlabeled_sets = relabel_dataset(img_dir, labels)
        data_dict = dict()
        data_dict['labeled'] = labeled_sets
        data_dict['unlabeld'] = unlabeled_sets

        pickle.dump(data_dict, open(args.cached_data_file, "wb"))
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        preset_data = pickle.load(open(args.cached_data_file, "rb"))
        labeled_sets= preset_data['labeled']
        unlabeled_sets = preset_data['unlabeld']
        print('load preset file')


    ####################### Make data loader #######################################

    train_transformation = TransformTwice(transforms.Compose([
        RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    labeled_data = Cifar_L(labeled_sets, train_transformation)
    unlabeled_data = Cifar_UL(unlabeled_sets,train_transformation)

    labeled_loader = data.DataLoader(labeled_data,
                                   batch_size=args.labeled_batchsize,
                                   shuffle=True,
                                   num_workers=args.workers)
    unlabeled_loader = data.DataLoader(unlabeled_data,
                                     batch_size=args.unlabeled_batchsize,
                                     shuffle=True,
                                     num_workers=args.workers)

    iter_unlabeled = iter(unlabeled_loader)
    dataloader = datasets.CIFAR10
    testset = dataloader(root=os.path.join(args.dataset, 'workdir'), train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers)

    ###########################  Model setting   ################################################
    model_C = cifar_shakeshake26(supervised= True)
    model_U = cifar_shakeshake26(supervised= False)

    optimizer_C = torch.optim.SGD(model_C.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    optimizer_U = torch.optim.SGD(model_U.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  nesterov=args.nesterov)

    class_criterion = nn.CrossEntropyLoss()
    recon_criterion = nn.MSELoss()
    Best_prec=0

    ##########################  CUDA setting ####################################################
    if use_cuda:
        model_C = model_C.cuda()
        model_U = model_U.cuda()
        class_criterion = class_criterion.cuda()
        recon_criterion = recon_criterion.cuda()

    ######################################################################################3

    for epoch in range(args.num_epochs):
        SP_lossC, SP_lossU, UnSP_lossC, UnSP_lossU = 0, 0, 0, 0
        start_time = time.time()
        for i, (imgs, img2, labels) in enumerate(labeled_loader):
            try:
                imgs_ul, _= next(iter_unlabeled)
            except StopIteration:
                iter_unlabeled = iter(unlabeled_loader)
                imgs_ul, _ = next(iter_unlabeled)

            if use_cuda:
                imgs = imgs.cuda()
                img2 = img2.cuda()
                imgs_ul = imgs_ul.cuda()
                labels = labels.cuda()

            imgs = Variable(imgs)
            img2 = Variable(img2)
            imgs_ul = Variable(imgs_ul)
            labels = Variable(labels)
            ####################### sp train  dataset ##########################################

            Chat_x, Ch1, Ch2, Ch3, Cdnc_h1, Cdnc_h2, Cdnc_h3, Ch, Chat_y = model_C(imgs, True)
            _, _, _, _, _, _, _, _, Chat_y2 = model_C(img2, True)
            Uhat_x, Uh1, Uh2, Uh3, Udnc_h1, Udnc_h2, Udnc_h3, Uh, Uhat_y = model_U(imgs, False)

            for param in model_C.fc.parameters():
                param.requires_grad = True
            class_loss = class_criterion(Chat_y, labels)*args.lamdaC
            A= Chat_y2.detach()
            stability_loss = recon_criterion(Chat_y, A)*args.lamdaS

            Crecon_inter1= recon_criterion(Cdnc_h1, Ch1.detach())*args.lamdaL
            Crecon_inter2= recon_criterion(Cdnc_h2, Ch2.detach())*args.lamdaL
            Crecon_inter3= recon_criterion(Cdnc_h3, Ch3.detach())*args.lamdaL

            Urecon_inter1 = recon_criterion(Udnc_h1, Uh1.detach())*args.lamdaL
            Urecon_inter2 = recon_criterion(Udnc_h2, Uh2.detach())*args.lamdaL
            Urecon_inter3 = recon_criterion(Udnc_h3, Uh3.detach())*args.lamdaL

            lossC = class_loss + stability_loss + Crecon_inter1 + Crecon_inter2 + Crecon_inter3
            lossU = Urecon_inter1 + Urecon_inter2 + Urecon_inter3

            C_recon = recon_criterion(Chat_x, imgs)
            U_recon = recon_criterion(Uhat_x, imgs)

            if use_cuda:
                flag = (C_recon>U_recon).data.cpu()
            else:
                flag = (C_recon>U_recon).data

            if flag.numpy():
                # print("lossU ----sp")
                A= (recon_criterion(0.5*Chat_x.detach() + 0.5*Uhat_x, imgs)*args.lamdaR)
                lossU += (recon_criterion(0.5*Chat_x.detach() + 0.5*Uhat_x, imgs)*args.lamdaR)
            else:
                B=  (recon_criterion(0.5*Chat_x + 0.5*Uhat_x.detach(), imgs)*args.lamdaR)
                lossC += (recon_criterion(0.5*Chat_x + 0.5*Uhat_x.detach(), imgs)*args.lamdaR)

            SP_lossC += lossC.data[0]
            SP_lossU += lossU.data[0]

            optimizer_C.zero_grad()
            lossC.backward()
            optimizer_C.step()

            optimizer_U.zero_grad()
            lossU.backward()
            optimizer_U.step()


            ######################## unsp train dataset #########################################

            Chat_x, Ch1, Ch2, Ch3, Cdnc_h1, Cdnc_h2, Cdnc_h3, Ch, Chat_y = model_C(imgs_ul, False)
            Uhat_x, Uh1, Uh2, Uh3, Udnc_h1, Udnc_h2, Udnc_h3, Uh, Uhat_y = model_U(imgs_ul, False)

            for param in model_C.fc.parameters():
                param.requires_grad = False

            Crecon_inter1 = recon_criterion(Cdnc_h1, Ch1.detach()) * args.lamdaL
            Crecon_inter2 = recon_criterion(Cdnc_h2, Ch2.detach()) * args.lamdaL
            Crecon_inter3 = recon_criterion(Cdnc_h3, Ch3.detach()) * args.lamdaL

            Urecon_inter1 = recon_criterion(Udnc_h1, Uh1.detach()) * args.lamdaL
            Urecon_inter2 = recon_criterion(Udnc_h2, Uh2.detach()) * args.lamdaL
            Urecon_inter3 = recon_criterion(Udnc_h3, Uh3.detach()) * args.lamdaL

            lossC = Crecon_inter1 + Crecon_inter2 + Crecon_inter3
            lossU = Urecon_inter1 + Urecon_inter2 + Urecon_inter3

            C_recon = recon_criterion(Chat_x, imgs_ul)
            U_recon = recon_criterion(Uhat_x, imgs_ul)

            if use_cuda:
                flag = (C_recon>U_recon).data.cpu()
            else:
                flag = (C_recon>U_recon).data

            if flag.numpy():
                A= (recon_criterion(0.5*Chat_x.detach() + 0.5*Uhat_x, imgs_ul)*args.lamdaR)
                lossU += (recon_criterion(0.5*Chat_x.detach() + 0.5*Uhat_x, imgs_ul)*args.lamdaR)
            else:
                B = (recon_criterion(0.5*Chat_x + 0.5*Uhat_x.detach(), imgs_ul)*args.lamdaR)
                lossC += (recon_criterion(0.5*Chat_x + 0.5*Uhat_x.detach(), imgs_ul)*args.lamdaR)

            UnSP_lossC += lossC.data[0]
            UnSP_lossU += lossU.data[0]

            optimizer_C.zero_grad()
            lossC.backward()
            optimizer_C.step()

            optimizer_U.zero_grad()
            lossU.backward()
            optimizer_U.step()


        ######################### Test ############################################
        Test_loss, Top1, Top5 = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(testloader):

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            _, _, _, _, _, _, _, _, outputs = model_C(inputs, True)
            loss = class_criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            Test_loss += loss.data[0]
            Top1 += prec1[0]
            Top5 += prec5[0]

        ################## show info and save ##################################

        print("iter {} /{} : SP_dataset lossC = {:.3f}, SP_dataset lossU = {:.3f}, "
              "UnSP_dataset lossC = {:.3f}, UnSP_dataset lossU = {:.3f},"
              "Test_loss = {:.3f}, Top1_Acc = {:.2f}, Top5_Acc = {:.2f}, time = {:.2f}".format(
            epoch, args.num_epochs, SP_lossC/len(labeled_loader), SP_lossU/len(labeled_loader),
            UnSP_lossC/len(unlabeled_loader), UnSP_lossU/len(unlabeled_loader),
            Test_loss/len(testloader), Top1/len(testloader), Top5/len(testloader), time.time() - start_time))

        if Best_prec < Top1/len(testloader):
            torch.save(model_C.state_dict(),os.path.join(save_dir, str(epoch) + '_Model_C.pth'))
            torch.save(model_U.state_dict(),os.path.join(save_dir, str(epoch) + '_Model_U.pth'))
