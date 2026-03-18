import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Text CNN model
class textCNN(nn.Module):
    
    def __init__(self, vocab_built, dim_channel, kernel_wins, dropout_rate, num_class):
        super(textCNN, self).__init__()
        #load pretrained embedding in embedding layer.
        emb_dim = vocab_built.vectors.size()[1]
        self.embed = nn.Embedding(len(vocab_built), emb_dim)
        self.embed.weight.data.copy_(vocab_built.vectors)
        self.max_kernel_size = max(kernel_wins)

        #Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        #Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        #FC layer
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)

    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)

        # ==================== 【新增 2】：动态填充逻辑 ====================
        # 获取当前 batch 中句子的真实长度 (seq_len)
        seq_len = emb_x.size(2)

        # 如果句子长度小于最大的卷积核尺寸，则在句子末尾补 0
        if seq_len < self.max_kernel_size:
            pad_size = self.max_kernel_size - seq_len
            # F.pad 的补齐规则是从最后面的维度开始往前推：
            # (左侧补0, 右侧补0, 上方补0, 下方补pad_size)
            # 对应到这里就是：词向量维度不补，句子长度维度在末尾补齐
            emb_x = F.pad(emb_x, (0, 0, 0, pad_size))
        # =================================================================

        con_x = [conv(emb_x) for conv in self.convs]

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]

        fc_x = torch.cat(pool_x, dim=1)

        fc_x = fc_x.squeeze(-1)

        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return logit
