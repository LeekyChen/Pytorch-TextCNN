# -*- coding: utf-8 -*-
import argparse
import torch
import torchtext.data
from torchtext.data import Field, LabelField, TabularDataset, Iterator

# 导入你写好的其他模块
import dataset
import model
import training


def main():
    parser = argparse.ArgumentParser(description='Evaluate TextCNN on new dataset')

    # 核心路径参数
    parser.add_argument("--dataset_path", type=str,
                        help="需要测试的新数据集CSV路径",
                        default=r"./dataset/dataset_Davidson_sampled_seed42.csv")
    parser.add_argument("--model_path", type=str, default="best_validation",
                        help="训练好的模型权重路径")
    parser.add_argument("--vocab_path", type=str, default="text_vocab.pt",
                        help="训练时保存的词典路径")

    # 必须和训练时保持一致的模型参数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--kernel-height', type=str, default='3,4,5')
    parser.add_argument('--out-channel', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num-class', type=int, default=2)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("准备开始评估新数据集...")
    print(f"Dataset: {args.dataset_path}")

    # ================= 1. 加载并重建环境 =================
    print("Loading vocabulary...")
    # 加载老词典
    saved_vocab = torch.load(args.vocab_path)

    # 重新定义 Field
    tokenizer = torchtext.data.get_tokenizer('basic_english')
    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = LabelField(dtype=torch.long)

    TEXT.preprocessing = torchtext.data.Pipeline(dataset.clean_str)

    # ================= 2. 加载新数据集 =================
    print("Loading new tabular data...")
    my_datafields = [('comment_id', None), ('text', TEXT), ('label', LABEL)]

    new_tabular_data = TabularDataset(
        path=args.dataset_path,
        format='csv',
        skip_header=True,
        fields=my_datafields
    )

    # 【核心！】把老词典强行塞给当前的 TEXT
    TEXT.vocab = saved_vocab

    # 手动重建老标签映射 (确保 0 对应 not_hate, 1 对应 hate)
    LABEL.build_vocab(new_tabular_data)
    LABEL.vocab.stoi.update({'not_hate': 0, 'hate': 1})  # 强行锁定索引，防止反转

    # 创建迭代器
    new_data_iter = Iterator(
        new_tabular_data,
        batch_size=args.batch_size,
        device=device,
        sort_within_batch=False,
        repeat=False,
        train=False,  # 关闭 shuffle
        sort=False,
        sort_key=lambda x: len(x.text),
        shuffle=False
    )

    # ================= 3. 加载模型并测试 =================
    print("Loading model weights...")
    kernels = [int(x) for x in args.kernel_height.split(',')]
    m = model.textCNN(saved_vocab, args.out_channel, kernels, args.dropout, args.num_class).to(device)

    # 灌入权重参数
    m.load_state_dict(torch.load(args.model_path))

    print("\n" + "=" * 45)
    print("🚀 Running Evaluation...")
    # 调用你刚才在 training.py 里写好的豪华版 test 方法
    final_loss, final_acc, f1_hate, f1_nothate, macro_f1 = training.test(m, device, new_data_iter)

    print(f"📌 New Dataset Loss:      {final_loss:.4f}")
    print(f"📊 Accuracy:              {final_acc:.2f}%")
    print(f"🔥 F1_hate (Class 1):     {f1_hate:.2f}%")
    print(f"🕊️  F1_nothate (Class 0):  {f1_nothate:.2f}%")
    print(f"🌟 Macro_F1:              {macro_f1:.2f}%")
    print("=" * 45 + "\n")


if __name__ == '__main__':
    main()