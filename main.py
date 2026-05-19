# -*- coding: utf-8 -*-
"""main.ipynb

@author: HSU, CHIH-CHAO
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import argparse

import torch
import torch.optim as optim

import dataset
import model
import training

import matplotlib.pyplot as plt



#%%

def main():
    
    print("Pytorch Version:", torch.__version__)
    parser = argparse.ArgumentParser(description='TextCNN')
    #Training args
    parser.add_argument('--data-csv', type=str, default='./IMDB_Dataset.csv',
                        help='file path of training data in CSV format (default: ./train.csv)')
    
    parser.add_argument('--spacy-lang', type=str, default='en', 
                        help='language choice for spacy to tokenize the text')
                        
    parser.add_argument('--pretrained', type=str, default='glove.6B.300d',
                    help='choice of pretrined word embedding from torchtext')              
                        
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training (default: 64)')
    
    parser.add_argument('--val-batch-size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    
    parser.add_argument('--kernel-height', type=str, default='3,4,5',
                    help='how many kernel width for convolution (default: 3, 4, 5)')
    
    parser.add_argument('--out-channel', type=int, default=100,
                    help='output channel for convolutionaly layer (default: 100)')
    
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate for linear layer (default: 0.5)')
    
    parser.add_argument('--num-class', type=int, default=2,
                        help='number of category to classify (default: 2)')
    
    #if you are using jupyternotebook with argparser
    args = parser.parse_known_args()[0]
    #args = parser.parse_args()
    
    
    #Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    #%% Split whole dataset into train and valid set
    dataset.split_train_valid_test(args.data_csv, './train.csv', './valid.csv', './test.csv', 0.7)
    
    trainset, validset, testset, vocab = dataset.create_tabular_dataset('./train.csv',
                                 './valid.csv', './test.csv', args.spacy_lang, args.pretrained)
    torch.save(vocab, "text_vocab.pt")
    
    #%%Show some example to show the dataset
    print("Show some examples from train/valid..")
    print(trainset[0].text,  trainset[0].label)
    print(validset[0].text,  validset[0].label)
    
    train_iter, valid_iter, test_iter = dataset.create_data_iterator(args.batch_size, args.val_batch_size,
                                                         trainset, validset, testset, device)
                
    #%%Create
    kernels = [int(x) for x in args.kernel_height.split(',')]
    m = model.textCNN(vocab, args.out_channel, kernels, args.dropout , args.num_class).to(device)
    # print the model summery
    print(m)    
        
    train_loss = []
    train_acc = []
    val_losses = []
    val_accs = []
    best_val_acc = -1
    
    #optimizer
    optimizer = optim.Adam(m.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs+1):
        #train loss
        tr_loss, tr_acc = training.train(m, device, train_iter, optimizer, epoch, args.epochs)
        print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc))
        
        val_loss, val_acc = training.valid(m, device, valid_iter)
        print('Valid Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, val_loss, val_acc))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            #save paras(snapshot)
            print("model saves at {}% accuracy".format(best_val_acc))
            torch.save(m.state_dict(), "best_validation")
            
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    # === 终极考核环节：加载最好模型跑测试集 ===
    print("\n" + "=" * 45)
    print("🚀 Training finished! Loading best model for Test Set...")

    # 加载保存在硬盘上的最佳验证集模型权重
    m.load_state_dict(torch.load("best_validation"))

    # 调用我们刚刚重写的 test 方法
    final_loss, final_acc, f1_hate, f1_nothate, macro_f1, precision, recall, fpr = training.test(m, device, test_iter)

    print(f"📌 Final Test Loss:       {final_loss}")
    print(f"📊 Accuracy:              {final_acc}")
    print(f"🎯 Precision:             {precision}")
    print(f"📞 Recall:                {recall}")
    print(f"🔥 F1_hate (Class 1):     {f1_hate}")
    print(f"🕊️  F1_nothate (Class 0):  {f1_nothate}")
    print(f"🌟 Macro_F1:              {macro_f1}")
    print(f"⚠️  FPR (False Positive):  {fpr}")
    print("=" * 45 + "\n")

    
    #plot train/validation loss versus epoch
    #plot train/validation loss versus epoch
    x = list(range(1, args.epochs+1))
    plt.figure()
    plt.title("train/validation loss versus epoch")
    plt.xlabel("epoch")
    plt.ylabel("Average loss")
    plt.plot(x, train_loss,label="train loss")
    plt.plot(x, val_losses, color='red', label="val loss")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    #plot train/validation accuracy versus epoch
    x = list(range(1, args.epochs+1))
    plt.figure()
    plt.title("train/validation accuracy versus epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.plot(x, train_acc,label="train accuracy")
    plt.plot(x, val_accs, color='red', label="val accuracy")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

"""
训练:
python main.py --data-csv ./dataset/dataset_MHS.csv
=============================================
🚀 Training finished! Loading best model for Test Set...
📌 Final Test Loss:       0.030649544493570588
📊 Accuracy:              0.8183996166746527
🎯 Precision:             0.7998472116119175
📞 Recall:                0.6785482825664291
🔥 F1_hate (Class 1):     0.7342215988779803
🕊️  F1_nothate (Class 0):  0.8620815138282387
🌟 Macro_F1:              0.7981515563531095
⚠️  FPR (False Positive):  0.09958190801976435
=============================================


预测:
直接运行test_model.py 
Davidson:
=============================================
🚀 Running Evaluation...
📌 Final Test Loss:       0.050301657515051286
📊 Accuracy:              0.7418201323082424
🎯 Precision:             0.49645390070921985
📞 Recall:                0.6853146853146853
🔥 F1_hate (Class 1):     0.5757931844888367
🕊️  F1_nothate (Class 0):  0.8144435877666409
🌟 Macro_F1:              0.6951183861277388
⚠️  FPR (False Positive):  0.23877011770357914
=============================================

CHSD
=============================================
🚀 Running Evaluation...
📌 Final Test Loss:       0.04649527849753698
📊 Accuracy:              0.5828333333333333
🎯 Precision:             0.46994535519125685
📞 Recall:                0.03451043338683788
🔥 F1_hate (Class 1):     0.06429906542056076
🕊️  F1_nothate (Class 0):  0.7315817694369974
🌟 Macro_F1:              0.39794041742877906
⚠️  FPR (False Positive):  0.02765108323831243
=============================================


implicit:
=============================================
🚀 Running Evaluation...
📌 Final Test Loss:       0.06831517901023229
📊 Accuracy:              0.6531666666666667
🎯 Precision:             0.5048899755501223
📞 Recall:                0.1977022498803255
🔥 F1_hate (Class 1):     0.2841417268661851
🕊️  F1_nothate (Class 0):  0.7711426371934456
🌟 Macro_F1:              0.5276421820298154
⚠️  FPR (False Positive):  0.10355407824085912
=============================================

"""

# 效果只能说很拉。。。我猜是分词器的锅

# 没有测试集只有验证集？？？