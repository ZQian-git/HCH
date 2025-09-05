import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from evaluate import calculate_top_map,calculate_pr
from load_dataset import load_dataset
from metric import ContrastiveLoss
from model import FuseTransEncoder, ImageMlp, TextMlp
from os import path as osp

from HGCN import HGCN
from HGCN.hypergraph_utils import construct_H_with_KNN_from_distance, generate_G_from_H
from utils import load_checkpoints, save_checkpoints
from plot_pr_curve import plot_pr
from torch.optim import lr_scheduler

import time


class Solver(object):
    def __init__(self, config):
        self.config = config
        self.batch_size = 128
        self.total_epoch = config.epoch  # 默认50
        self.dataset = config.dataset  # 默认mirflickr
        self.model_dir = "./checkpoints"
        self.best_it = 0.0
        self.best_ti = 0.0

        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device(config.device if USE_CUDA else "cpu")

        self.task = config.task
        self.feat_lens = 512  # feature_length
        self.nbits = config.hash_lens  # 哈希码长度，默认为16
        self.topk = config.topk
        num_layers, self.token_size, nhead = 2, 1024, 4

        self.FuseTrans = FuseTransEncoder(num_layers, self.token_size, nhead).to(self.device)
        self.ImageMlp = ImageMlp(self.feat_lens, self.nbits).to(self.device)
        self.TextMlp = TextMlp(self.feat_lens, self.nbits).to(self.device)
        self.ImageHGCN = HGCN(self.feat_lens, self.nbits).to(self.device)
        self.TextHGCN = HGCN(self.feat_lens, self.nbits).to(self.device)

        paramsFuse_to_update = list(self.FuseTrans.parameters())
        paramsImage = list(self.ImageMlp.parameters())
        paramsText = list(self.TextMlp.parameters())
        paramsHGCNImage = list(self.ImageHGCN.parameters())
        paramsHGCNText = list(self.TextHGCN.parameters())

        # 计算参数的总数量
        total_param = sum([param.nelement() for param in paramsFuse_to_update]) + sum(
            [param.nelement() for param in paramsImage]) + sum([param.nelement() for param in paramsText]) + \
                      sum([param.nelement() for param in paramsHGCNImage]) + sum(
            [param.nelement() for param in paramsHGCNText])

        print("total_param:", total_param)

        self.optimizer_FuseTrans = optim.Adam(paramsFuse_to_update, lr=1e-4, betas=(0.5, 0.999))
        self.optimizer_ImageMlp = optim.Adam(paramsImage, lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_TextMlp = optim.Adam(paramsText, lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_ImageHGCN = optim.Adam(paramsHGCNImage, lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_TextHGCN = optim.Adam(paramsHGCNText, lr=1e-3, betas=(0.5, 0.999))

        if self.dataset == "mirflickr" or self.dataset == "nus-wide":
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, milestones=[30, 80], gamma=1.2)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, milestones=[30, 80], gamma=1.2)
            self.ImageHGCN_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageHGCN, milestones=[30, 80],
                                                               gamma=1.2)
            self.TextHGCN_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextHGCN, milestones=[30, 80], gamma=1.2)
        elif self.dataset == "mscoco":
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, milestones=[200], gamma=0.6)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, milestones=[200], gamma=0.6)
            self.ImageHGCN_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageHGCN, milestones=[200], gamma=0.6)
            self.TextHGCN_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextHGCN, milestones=[200], gamma=0.6)

        data_loader = load_dataset(self.dataset, self.batch_size)
        self.train_loader = data_loader['train']
        self.query_loader = data_loader['query']
        self.retrieval_loader = data_loader['retrieval']

        # 创建对比损失
        self.ContrastiveLoss = ContrastiveLoss(batch_size=self.batch_size, device=self.device)

    def train(self):
        if self.task == 0:
            print("Training Hash Fuction...")
            I2T_MAP = []  # image to text
            T2I_MAP = []
            LossList = []
            for epoch in range(self.total_epoch):
                print("epoch:", epoch + 1)
                train_loss = self.trainhash(epoch)  # 训练哈希函数
                if epoch < 100:
                    LossList.append(train_loss)
                print(train_loss)
                if ((epoch + 1) % 10 == 0) or epoch == 0:
                    print("Testing...")
                    img2text, text2img = self.evaluate()
                    I2T_MAP.append(img2text)
                    T2I_MAP.append(text2img)
                    print('I2T:', img2text, ', T2I:', text2img)
                    print('bestI2T:', self.best_it, ', bestT2I:', self.best_ti)

        return (self.best_it + self.best_ti) / 2., self.best_it, self.best_ti  # 返回各自的map以及两者平均的map

    def evaluate(self):
        pass

    def trainhash(self, epoch):
        self.FuseTrans.train()
        self.ImageMlp.train()
        self.TextMlp.train()
        self.ImageHGCN.train()
        self.TextHGCN.train()
        running_loss = 0.0
        self.ImageHGCN.set_alpha(epoch)
        self.TextHGCN.set_alpha(epoch)
        self.ImageMlp.set_alpha(epoch)
        self.TextMlp.set_alpha(epoch)

        for idx, (img, txt, _, _) in enumerate(self.train_loader):
            img, txt = img.to(self.device), txt.to(self.device)
            temp_tokens = torch.concat((img, txt), dim=1).unsqueeze(0)
            img_embedding, text_embedding = self.FuseTrans(temp_tokens)

            img = F.normalize(img)
            txt = F.normalize(txt)
            S1 = self.cal_similarity_matrix(img, txt)
            S2 = self.cal_similarity_matrix(img_embedding, text_embedding)
            S = S1 * self.config.alpha_S + S2 * (1 - self.config.alpha_S)

            code_I = self.ImageMlp(img_embedding * (1 - self.config.alpha_I) + img * self.config.alpha_I)
            code_T = self.TextMlp(text_embedding * (1 - self.config.alpha_T) + txt * self.config.alpha_T)
            loss1 = self.ContrastiveLoss(img_embedding, text_embedding)
            loss2 = self.ContrastiveLoss(code_I, code_T)

            H = construct_H_with_KNN_from_distance(S, device=self.device)
            G = generate_G_from_H(H, self.device)
            code_GI = self.ImageHGCN(img_embedding * (1 - self.config.alpha_I) + img * self.config.alpha_I, G)
            code_GT = self.TextHGCN(text_embedding * (1 - self.config.alpha_T) + txt * self.config.alpha_T, G)
            loss3 = self.cal_loss(code_I, code_T, code_GI, code_GT, S)

            loss = self.config.lambda1 * loss1 + self.config.lambda2 * loss2 + self.config.lambda3 * loss3

            self.optimizer_FuseTrans.zero_grad()
            self.optimizer_ImageMlp.zero_grad()
            self.optimizer_TextMlp.zero_grad()
            self.optimizer_TextHGCN.zero_grad()
            self.optimizer_ImageHGCN.zero_grad()
            loss.backward()
            self.optimizer_FuseTrans.step()
            self.optimizer_ImageMlp.step()
            self.optimizer_TextMlp.step()
            self.optimizer_TextHGCN.step()
            self.optimizer_ImageHGCN.step()
            running_loss += loss.item()

            self.ImageMlp_scheduler.step()
            self.TextMlp_scheduler.step()
            self.ImageHGCN_scheduler.step()
            self.TextHGCN_scheduler.step()

        return running_loss



    def cal_similarity_matrix(self, F_I, F_T):
         pass


    def cal_loss(self, code_I, code_T, code_GI, code_GT, S):
        pass
