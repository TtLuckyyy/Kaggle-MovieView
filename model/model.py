import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Mymodel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, n_class, bidirection=True, num_heads=8,
                 num_transformer_layers=4):
        super(Mymodel, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.bidirection = bidirection
        self.embedding = torch.nn.Embedding(input_size, hidden_size)

        # 因为此时仅仅需要提取特征，所以这里只使用了transformer的编码层
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True),
            num_layers=num_transformer_layers
        )

        # LSTM Layer
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, n_layer, batch_first=True, bidirectional=bidirection)

        # Fully Connected Layer
        self.fc = torch.nn.Linear(hidden_size * (2 if bidirection else 1), n_class)

    def forward(self, x, seq_len):

        # 先将输入的x通过embedding层进行嵌入
        x = self.embedding(x)

        # 先通过 Transformer 进行编码
        x = self.transformer(x)

        # Packing 输入序列
        packed_input = pack_padded_sequence(x, seq_len.cpu(), batch_first=True)

        # 获取 packed_input 所在的设备
        device = packed_input.data.device

        # 初始化 LSTM 隐藏状态和细胞状态
        h0 = torch.zeros(self.n_layer * (2 if self.bidirection else 1), x.size(0), self.hidden_size)
        c0 = torch.zeros(self.n_layer * (2 if self.bidirection else 1), x.size(0), self.hidden_size)

        # 将 h0 和 c0 移动到相同的设备
        h0 = h0.to(device)
        c0 = c0.to(device)

        # 通过 LSTM
        packed_out, _ = self.lstm(packed_input, (h0.detach(), c0.detach()))

        # 将输出的 packed_out 展平
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # 解包

        # 选择每个序列的最后一个时间步的输出
        # 之所以是seq_len-1，是因为pack_padded_sequence的作用，将长度小于等于max_length的序列进行了padding，
        # 所以实际的序列长度是seq_len-1，最后一个时间步的输出即为序列的最后一个词的输出,而不是max_length-1
        out = out[torch.arange(out.size(0)), seq_len - 1, :]

        # 全连接层
        out = self.fc(out)
        return out