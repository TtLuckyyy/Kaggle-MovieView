# 训练集验证集数据对象
import math
from itertools import chain
from config import INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE, N_LAYER, N_EPOCHS, USE_GPU
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from model.model import Mymodel
from 深度学习.MovieView.utils.DataProcess import MovieDataset
from 深度学习.MovieView.utils.Totensor import make_tensor, make_tensor_test

train_set = MovieDataset(train=True, max_length=128)  # 设置最大文本长度为128
validation_set = MovieDataset(train=False, max_length=128)

# 超参数
N_CLASS = len(set(train_set.y_data))

# 设定训练集和验证集
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)


# 模型训练
def trainModel():
    total_loss = 0
    for i, (phrase, sentiment) in enumerate(train_loader, 0):
        inputs, seq_lengths, target = make_tensor(phrase, sentiment)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{i * len(inputs)}/{len(train_set)}]', end='')
            print(f'loss={total_loss / ((i+1) * len(inputs))}')

# 验证集评估
def evalModel():
    correct = 0
    total = len(validation_set)
    print("Evaluating trained model...")
    with torch.no_grad():
        for i, (phrase, sentiment) in enumerate(validation_loader, 0):
            inputs, seq_lengths, target = make_tensor(phrase, sentiment)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Validation set: Accuracy {correct}/{total} {percent}%')
    return correct / total


# 获取测试集
def get_test_set():
    test_set = pd.read_csv('data/test_part.tsv', sep='\t')  # 使用关键字参数 sep
    PhraseId = test_set['PhraseId']
    Phrase = test_set['Phrase']
    return PhraseId, Phrase

# 使用测试集进行测试
def predict():
    PhraseId, Phrase = get_test_set()  # 获取测试集
    sentiment_list = []  # 定义预测结果列表
    batchNum = math.ceil(PhraseId.shape[0] / BATCH_SIZE)  # 获取总的Batch数
    classifier = torch.load('./weight/sentimentAnalyst.pkl') # 注意，这里的权重文件是在Cuda上训练出来的，因此只可以使用GPU进行预测

    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)
    else:
        device = torch.device("cpu")


    with torch.no_grad():
        for i in range(batchNum):
            if i == batchNum - 1:
                phraseBatch = Phrase[BATCH_SIZE * i:]  # 处理最后不足BATCH_SIZE的情况
            else:
                phraseBatch = Phrase[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]

            inputs, seq_lengths, org_idx = make_tensor_test(phraseBatch)
            # 确保所有数据都在同一个设备上
            inputs, seq_lengths = inputs.to(device), seq_lengths.to(device)

            output = classifier(inputs, seq_lengths)
            sentiment = output.max(dim=1, keepdim=True)[1]
            sentiment = sentiment[org_idx].squeeze(1)
            sentiment_list.append(sentiment.cpu().numpy().tolist())

    sentiment_list = list(chain.from_iterable(sentiment_list))  # 将sentiment_list按行拼成一维列表
    result = pd.DataFrame({'PhraseId': PhraseId, 'Sentiment': sentiment_list})
    result.to_csv('./result.csv', index=False)  # 保存结果


if __name__ == '__main__':
    # *********************************** 训练模型 ****************************************************
    classifier = Mymodel(INPUT_SIZE, HIDDEN_SIZE, N_LAYER, N_CLASS, bidirection=False, num_heads=8,num_transformer_layers=4)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)
    else:
        device = torch.device("cpu")

    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        print("********************   Training for %d epochs   ***********************" % epoch)
        trainModel()
        acc = evalModel()
        acc_list.append(acc)

        # 保存最优时的模型
        if acc >= max(acc_list):
            torch.save(classifier, './weight/sentimentAnalyst.pkl')
            print('Save Model!')

    # 绘制训练准确度曲线
    epoch = [epoch + 1 for epoch in range(len(acc_list))]
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

    # *********************************** 预测模型 ****************************************************
    # 预测结果并保存到result.csv文件中
    # predict()