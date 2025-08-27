import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from evaluate import calculate_top_map
from load_dataset import load_dataset
from metric import ContrastiveLoss
from model import FuseTransEncoder, ImageMlp, TextMlp
from os import path as osp

from HGCN import HGCN
from HGCN.hypergraph_utils import construct_H_with_KNN_from_distance, generate_G_from_H
from utils import load_checkpoints, save_checkpoints
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
        elif self.task == 1:
            print("Training Hash Fuction without HGCN...")
            for epoch in range(self.total_epoch):
                print("epoch:", epoch + 1)
                train_loss = self.trainWithoutHGCN(epoch)  # 训练哈希函数
                print(train_loss)
                if ((epoch + 1) % 10 == 0):
                    print("Testing...")
                    img2text, text2img = self.evaluate()
                    print('I2T:', img2text, ', T2I:', text2img)
                    print('bestI2T:', self.best_it, ', bestT2I:', self.best_ti)
        elif self.task == 2:
            print("Training Hash Fuction without Transformer...")
            for epoch in range(self.total_epoch):
                print("epoch:", epoch + 1)
                train_loss = self.trainWithoutTransformer(epoch)  # 训练哈希函数
                print(train_loss)
                if ((epoch + 1) % 10 == 0):
                    print("Testing...")
                    img2text, text2img = self.evaluate()
                    print('I2T:', img2text, ', T2I:', text2img)
                    print('bestI2T:', self.best_it, ', bestT2I:', self.best_ti)
        elif self.task == 3:
            print("Training Hash Fuction without MMSE...")
            for epoch in range(self.total_epoch):
                print("epoch:", epoch + 1)
                train_loss = self.trainWithoutMMSE(epoch)  # 训练哈希函数
                print(train_loss)
                if ((epoch + 1) % 10 == 0):
                    print("Testing...")
                    img2text, text2img = self.evaluate()
                    print('I2T:', img2text, ', T2I:', text2img)
                    print('bestI2T:', self.best_it, ', bestT2I:', self.best_ti)


        # with open('results/Convergence/loss/' + self.dataset + 'Loss.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(LossList)
        #
        # with open('results/Convergence/map/' + self.dataset + 'Map.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(I2T_MAP)
        #     writer.writerow(T2I_MAP)
        return (self.best_it + self.best_ti) / 2., self.best_it, self.best_ti  # 返回各自的map以及两者平均的map

    def evaluate(self):
        self.FuseTrans.eval()
        self.ImageMlp.eval()
        self.TextMlp.eval()
        self.ImageHGCN.eval()
        self.TextHGCN.eval()
        qu_BI, qu_BT, qu_L = [], [], []  # query_image_bit ,query_text_bit,query_label
        re_BI, re_BT, re_L = [], [], []  # retrieval_image_bit ,retrieval_text_bit,retrieval_label

        with torch.no_grad():
            for _, (data_I, data_T, data_L, _) in enumerate(self.query_loader):
                data_I, data_T = data_I.to(self.device), data_T.to(self.device)
                if self.task in [0, 1, 3]:
                    temp_tokens = torch.concat((data_I, data_T), dim=1)
                    img_query, txt_query = self.FuseTrans(temp_tokens)

                    img_query = self.ImageMlp(
                        img_query * (1 - self.config.alpha_I) + F.normalize(data_I) * self.config.alpha_I)
                    txt_query = self.TextMlp(
                        txt_query * (1 - self.config.alpha_T) + F.normalize(data_T) * self.config.alpha_T)
                elif self.task in [2]:
                    img_query = self.ImageMlp(F.normalize(data_I))
                    txt_query = self.TextMlp(F.normalize(data_T))

                img_query, txt_query = img_query.cpu().numpy(), txt_query.cpu().numpy()
                qu_BI.extend(img_query)
                qu_BT.extend(txt_query)
                qu_L.extend(data_L.cpu().numpy())

            for _, (data_I, data_T, data_L, _) in enumerate(self.retrieval_loader):
                data_I, data_T = data_I.to(self.device), data_T.to(self.device)
                if self.task in [0, 1, 3]:
                    temp_tokens = torch.concat((data_I, data_T), dim=1)
                    img_retrieval, txt_retrieval = self.FuseTrans(temp_tokens)

                    img_retrieval = self.ImageMlp(
                        img_retrieval * (1 - self.config.alpha_I) + F.normalize(data_I) * self.config.alpha_I)
                    txt_retrieval = self.TextMlp(
                        txt_retrieval * (1 - self.config.alpha_T) + F.normalize(data_T) * self.config.alpha_T)
                elif self.task in [2]:

                    img_retrieval = self.ImageMlp(F.normalize(data_I))
                    txt_retrieval = self.TextMlp(F.normalize(data_T))

                img_retrieval, txt_retrieval = img_retrieval.cpu().numpy(), txt_retrieval.cpu().numpy()
                re_BI.extend(img_retrieval)
                re_BT.extend(txt_retrieval)
                re_L.extend(data_L.cpu().numpy())

        re_BI = np.array(re_BI)
        re_BT = np.array(re_BT)
        re_L = np.array(re_L)

        qu_BI = np.array(qu_BI)
        qu_BT = np.array(qu_BT)
        qu_L = np.array(qu_L)

        qu_BI = torch.sign(torch.tensor(qu_BI)).cpu().numpy()
        qu_BT = torch.sign(torch.tensor(qu_BT)).cpu().numpy()
        re_BT = torch.sign(torch.tensor(re_BT)).cpu().numpy()
        re_BI = torch.sign(torch.tensor(re_BI)).cpu().numpy()

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=self.topk)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=self.topk)

        if (self.best_it + self.best_ti) < (MAP_I2T + MAP_T2I):
            self.best_it = MAP_I2T
            self.best_ti = MAP_T2I
        return MAP_I2T, MAP_T2I

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

    def trainWithoutHGCN(self, epoch):
        self.FuseTrans.train()
        self.ImageMlp.train()
        self.TextMlp.train()

        running_loss = 0.0
        self.ImageMlp.set_alpha(epoch)
        self.TextMlp.set_alpha(epoch)

        for idx, (img, txt, _, _) in enumerate(self.train_loader):
            img, txt = img.to(self.device), txt.to(self.device)
            temp_tokens = torch.concat((img, txt), dim=1)
            temp_tokens = temp_tokens.unsqueeze(0)
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
            loss3 = self.cal_loss_withoutHGCN(code_I, code_T, S)

            loss = self.config.lambda1 * loss1 + self.config.lambda2 * loss2 + self.config.lambda3 * loss3

            self.optimizer_FuseTrans.zero_grad()
            self.optimizer_ImageMlp.zero_grad()
            self.optimizer_TextMlp.zero_grad()

            loss.backward()
            self.optimizer_FuseTrans.step()
            self.optimizer_ImageMlp.step()
            self.optimizer_TextMlp.step()

            running_loss += loss.item()
            self.ImageMlp_scheduler.step()
            self.TextMlp_scheduler.step()

        return running_loss

    def trainWithoutTransformer(self, epoch):

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

            img = F.normalize(img)
            txt = F.normalize(txt)
            S = self.cal_similarity_matrix(img, txt)

            code_I = self.ImageMlp(img)
            code_T = self.TextMlp(txt)

            loss2 = self.ContrastiveLoss(code_I, code_T)

            H = construct_H_with_KNN_from_distance(S, device=self.device)
            G = generate_G_from_H(H, self.device)
            code_GI = self.ImageHGCN(img, G)
            code_GT = self.TextHGCN(txt, G)
            loss3 = self.cal_loss(code_I, code_T, code_GI, code_GT, S)

            loss = self.config.lambda2 * loss2 + self.config.lambda3 * loss3

            self.optimizer_ImageMlp.zero_grad()
            self.optimizer_TextMlp.zero_grad()
            self.optimizer_TextHGCN.zero_grad()
            self.optimizer_ImageHGCN.zero_grad()
            loss.backward()
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

    def trainWithoutMMSE(self, epoch):
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
            S1 = self.cal_similarity_matrix_no_enhancing(img, txt)
            S2 = self.cal_similarity_matrix_no_enhancing(img_embedding, text_embedding)
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

        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        S_I = S_I * 2 - 1

        F_T = F.normalize(F_T)
        S_T = F_T.mm(F_T.t())
        S_T = S_T * 2 - 1

        S_high = F.normalize(S_I).mm(F.normalize(S_T).t())

        S_ = self.config.w1 * S_I + self.config.w2 * S_T + self.config.w3 * (S_high + S_high.t()) / 2

        S_mean = torch.mean(S_).detach()
        S_min = torch.min(S_).detach()
        S_max = torch.max(S_).detach()
        left = S_mean
        right = S_mean

        S_[S_ < left] = (1 + 0.1 * torch.exp(-(S_[S_ < left] - S_min))) * S_[S_ < left]
        S_[S_ > right] = (1 + 0.2 * torch.exp(S_[S_ > right] - S_max)) * S_[S_ > right]

        S = S_ * 1.4

        return S

    def cal_similarity_matrix_no_enhancing(self, F_I, F_T):

        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        S_I = S_I * 2 - 1

        F_T = F.normalize(F_T)
        S_T = F_T.mm(F_T.t())
        S_T = S_T * 2 - 1

        S_high = F.normalize(S_I).mm(F.normalize(S_T).t())

        S = self.config.w1 * S_I + self.config.w2 * S_T + self.config.w3 * (S_high + S_high.t()) / 2



        return S


    def cal_loss(self, code_I, code_T, code_GI, code_GT, S):

        B_I = F.normalize(code_I, dim=1)
        B_T = F.normalize(code_T, dim=1)
        B_GI = F.normalize(code_GI, dim=1)
        B_GT = F.normalize(code_GT, dim=1)

        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())
        BT_BI = B_T.mm(B_I.t())
        GBI_GBT = B_GI.mm(B_GT.t())
        GI = B_GI.mm(B_I.t())
        GIT = B_GI.mm(B_T.t())

        loss1 = F.mse_loss(BI_BI, S) + F.mse_loss(BT_BT, S)
        loss2 = F.mse_loss(BI_BT, S) + F.mse_loss(BT_BI, S) - (B_I * B_T).sum(dim=1).mean()
        loss3 = F.mse_loss(GBI_GBT, S)
        loss4 = F.mse_loss(GI, S) + F.mse_loss(GIT, S)

        loss = self.config.beta1 * loss1 + self.config.beta2 * loss2 + self.config.beta3 * loss3 + self.config.beta4 * loss4
        return loss

    def cal_loss_withoutHGCN(self, code_I, code_T, S):

        B_I = F.normalize(code_I, dim=1)
        B_T = F.normalize(code_T, dim=1)

        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())
        BT_BI = B_T.mm(B_I.t())

        loss1 = F.mse_loss(BI_BI, S) + F.mse_loss(BT_BT, S)
        loss2 = F.mse_loss(BI_BT, S) + F.mse_loss(BT_BI, S) - (B_I * B_T).sum(dim=1).mean()

        loss = self.config.beta1 * loss1 + self.config.beta2 * loss2
        return loss
