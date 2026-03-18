# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:50:00 2019

@author: HSU, CHIH-CHAO
"""

#try to use nn for crossentropy

from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn.functional as F

#%% Training the Model
def train(m, device, train_itr, optimizer, epoch, max_epoch):
    m.train()
    corrects, train_loss = 0.0,0
    for batch in train_itr:
        text, target = batch.text, batch.label
        text = torch.transpose(text,0, 1)
        # target.data.sub_(1)
        text, target = text.to(device), target.to(device)
        optimizer.zero_grad()
        logit = m(text)

        loss = F.cross_entropy(logit, target)
        loss.backward()
        optimizer.step()

        train_loss+= loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    size = len(train_itr.dataset)
    train_loss /= size
    accuracy = 100.0 * corrects/size
  
    return train_loss, accuracy

def valid(m, device, test_itr):
    m.eval()
    corrects, test_loss = 0.0,0
    for batch in test_itr:
        text, target = batch.text, batch.label
        text = torch.transpose(text,0, 1)
        # target.data.sub_(1)
        text, target = text.to(device), target.to(device)

        logit = m(text)
        loss = F.cross_entropy(logit, target)


        test_loss += loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    size = len(test_itr.dataset)
    test_loss /= size 
    accuracy = 100.0 * corrects/size
    
    return test_loss, accuracy

def test(m, device, test_itr):
    m.eval()
    test_loss = 0.0

    # 用于收集整个测试集所有的预测值和真实值
    all_preds = []
    all_targets = []

    with torch.no_grad(): # 🔥 规范做法，测试时不计算梯度
        for batch in test_itr:
            text, target = batch.text, batch.label
            text = torch.transpose(text, 0, 1)
            text, target = text.to(device), target.to(device)

            logit = m(text)
            loss = F.cross_entropy(logit, target)
            test_loss += loss.item()

            # 获取预测的类别索引 (0 或 1)
            result = torch.max(logit, 1)[1]

            # 将张量转移到 CPU 并转为普通的 Python 列表，追加到总列表中
            all_preds.extend(result.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 计算平均 Loss
    size = len(test_itr.dataset)
    test_loss /= size

    # 🔥 1. 计算 Accuracy (乘以 100 转为百分比)
    accuracy = accuracy_score(all_targets, all_preds) * 100.0

    # 🔥 2. 计算每个单独类别的 F1
    # average=None 会严格按照标签索引 [0, 1] 的顺序返回一个数组
    # 因为我们已经确认了 0 是 not_hate, 1 是 hate：
    class_f1_scores = f1_score(all_targets, all_preds, average=None, labels=[0, 1])
    f1_nothate = class_f1_scores[0] * 100.0  # 索引 0
    f1_hate = class_f1_scores[1] * 100.0     # 索引 1

    # 🔥 3. 计算 Macro F1 (两类 F1 的宏平均)
    macro_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0

    # 将 5 个指标一起返回给主程序
    return test_loss, accuracy, f1_hate, f1_nothate, macro_f1