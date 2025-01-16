# Kaggle-MovieView
Kaggle平台MovieView竞赛的一个Baseline模型.
> 竞赛网址：(https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews)

```
markdown复制编辑# 情感分析模型（Sentiment Analysis Model）

该项目是一个基于深度学习的情感分析模型，使用了 Transformer 和 LSTM 结合的架构，旨在对电影评论数据进行情感分类（正面或负面）。本项目使用了 PyTorch 框架实现，包含了数据预处理、模型训练、评估和预测等模块，适合用于学习和研究 NLP 任务中的情感分析。


markdown复制编辑
## 环境要求

此项目使用 Python 3 及 PyTorch 框架开发，建议在虚拟环境中运行，并安装以下依赖：

- Python 3.6+
- PyTorch 1.8+
- pandas
- matplotlib
- scikit-learn

### 安装依赖

首先创建并激活虚拟环境：

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

然后安装项目所需的依赖：

```
bash


复制编辑
pip install -r requirements.txt
```

## 数据集

该项目使用电影评论数据集（来自 Kaggle 数据集），包括评论的短语（Phrase）和情感标签（Sentiment）。数据集已预处理并存储在 `data/` 文件夹下：

- `train.tsv`：包含训练数据。
- `test.tsv`：包含测试数据。

每行数据包含两个字段：

- `PhraseId`：评论的唯一标识符。
- `Phrase`：评论的内容。
- `Sentiment`：评论的情感标签（0 代表负面情感，1 代表正面情感，2）。

## 模型架构

本项目使用了 Transformer 和 LSTM 结合的混合模型架构，主要包括以下部分：

1. **Embedding层**：将输入的字符转化为固定维度的词向量。
2. **Transformer编码器**：提取输入序列的上下文信息。
3. **LSTM层**：进一步对序列信息进行处理，增强模型的记忆能力。
4. **全连接层**：输出情感预测结果。

模型的目标是对输入的电影评论进行情感分类，输出为 `0` 或 `1`。

## 训练与评估

### 训练模型

训练模型的代码位于 `main.py` 文件中。你可以通过以下命令开始训练：

```
bash


复制编辑
python main.py
```

训练过程中，模型会使用训练数据进行训练，并每10个批次输出一次损失值。

### 模型评估

训练完成后，你可以评估模型的性能，输出验证集的准确率。评估函数 `evalModel()` 会输出验证集的准确率。

### 保存模型

训练过程中，最佳的模型会被保存到 `weight/sentimentAnalyst.pkl` 文件中，你可以在训练完成后加载此模型进行预测。

## 预测

你可以使用训练好的模型对测试集进行情感预测，并将结果保存到 `result.csv` 文件中。预测函数 `predict()` 会根据输入的测试集数据生成预测结果。

运行以下命令进行预测：

```
bash


复制编辑
python main.py
```

预测结果会保存为 `result.csv` 文件，包含每个评论的 `PhraseId` 和对应的预测情感 `Sentiment`。

## 使用GPU加速

为了加快训练速度，你可以使用 GPU 进行训练。在 `config.py` 文件中设置 `USE_GPU = True` 来启用 GPU 训练。确保你的环境中已安装支持 CUDA 的 PyTorch 版本，并且有可用的 GPU。

## 文件说明

- `main.py`：包含模型的训练和推理过程。
- `Dataset_Process.py`：数据集加载和处理模块。
- `Totensor.py`：torch向量转换模块。
- `config.py`：全局参数定义模块。
- `sentimentAnalyst.pkl`：保存训练好的模型权重。
- `result.csv`：预测结果输出文件。

## 贡献

欢迎对该项目提出改进意见，提交 PR 或者报告问题。你可以通过以下方式联系我：

- Email: [3027943368@qq.com]

