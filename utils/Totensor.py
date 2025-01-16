import torch
import sys
import os
import sys
sys.path.append("../")
from config import USE_GPU

# 字符转ASCII码
def phrase2list(phrase):
    phrase = str(phrase)
    arr = [ord(c) for c in phrase]  # ord() 返回对应的ASCII码
    return arr, len(arr)

# 返回一个tensor，如果USE_GPU为True，则在GPU上创建，否则在CPU上创建
def create_tensor(tensor):
    if USE_GPU:
        device0 = torch.device('cuda:0')
        tensor = tensor.to(device0)
    else:
        device0 = torch.device('cpu')
        tensor = tensor.to(device0)
    return tensor


# 为训练集写的处理文本函数
def make_tensor(phrase, sentiment):
    # 获取短语对应的向量和所有短语的实际长度（为后面pack_up排序做准备）
    sequences_and_lengths = [phrase2list(phrase) for phrase in phrase]  # 名字字符串->字符数组->对应ASCII码
    phrase_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    sentiment = sentiment.long()
    # 先创造一个零向量矩阵，再将短语向量填充进去
    seq_tensor = torch.zeros(len(phrase_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(phrase_sequences, seq_lengths)):  # 填充零
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)  # name_sequences不够最大长度的位置补零

    # 排序 sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)  # perm_idx表示排完序元素原本的索引
    seq_tensor = seq_tensor[perm_idx]  # 对补零后的name_sequences按照长度排序
    sentiment = sentiment[perm_idx]  # 将短语对应的标签值进行重排
    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(sentiment)

# 为测试集写的处理文本函数
def make_tensor_test(phrase):
    sequences_and_lengths = [phrase2list(phrase) for phrase in phrase]  # 名字字符串->字符数组->对应ASCII码
    phrase_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])

    # make tensor of name, batchSize x seqLen
    seq_tensor = torch.zeros(len(phrase_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(phrase_sequences, seq_lengths)):  # 填充零
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)  # name_sequences不够最大长度的位置补零

    # 排序 sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)  # perm_idx表示排完序元素原本的索引
    seq_tensor = seq_tensor[perm_idx]  # 对补零后的name_sequences按照长度排序
    # 因为这里将测试集的每个Batch的文本顺序打乱了，记录原本的顺序org_idx，以便将预测出的结果顺序还原
    # 无论索引如何排列，这都是对原始矩阵的索引，所以只需要对他进行升序排序就可以得到原始矩阵
    _, org_idx = perm_idx.sort(descending=False)
    return create_tensor(seq_tensor), create_tensor(seq_lengths), org_idx

