import torch
from torch import nn
import math
from torch.autograd import Function
from torch.nn.utils import weight_norm
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import torch.nn.functional as F


# from utils import weights_init

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


##################################################
##########  BACKBONE NETWORKS  ###################
##################################################

########## CNN #############################

class CNNBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            # nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(output_channel),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # nn.Dropout(0.2)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.block1 = CNNBlock(configs.input_channels, configs.mid_channels, configs.kernel_size, configs.stride,
                               configs.padding)
        self.block2 = CNNBlock(configs.mid_channels, configs.mid_channels2, configs.kernel_size, configs.stride,
                               configs.padding)
        self.block3 = CNNBlock(configs.mid_channels2, configs.mid_channels3, configs.kernel_size, configs.stride,
                               configs.padding)
        self.block4 = CNNBlock(configs.mid_channels3, configs.mid_channels2, configs.kernel_size, configs.stride,
                               configs.padding)
        self.block5 = CNNBlock(configs.mid_channels2, configs.mid_channels, configs.kernel_size, configs.stride,
                               configs.padding)
        self.block6 = CNNBlock(configs.mid_channels, configs.final_out_channels, configs.kernel_size, configs.stride,
                               configs.padding)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

        # weights_init(self.conv_block1)
        # weights_init(self.conv_block2)
        # weights_init(self.conv_block3)

    def forward(self, x_in):
        x = self.block1(x_in)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat

class CNNT(nn.Module):
    def __init__(self, configs):
        super(CNNT, self).__init__()

        self.block1 = CNNBlock(configs.input_channels, configs.mid_channels, configs.kernel_size, configs.stride,
                               configs.padding)
        self.block2 = CNNBlock(configs.mid_channels, configs.mid_channels2, configs.kernel_size, configs.stride,
                               configs.padding)
        self.block3 = CNNBlock(configs.mid_channels2, configs.mid_channels3, configs.kernel_size, configs.stride,
                               configs.padding)
        self.block4 = CNNBlock(configs.mid_channels3, configs.mid_channels3, configs.kernel_size, configs.stride,
                               configs.padding)
        self.block5 = CNNBlock(configs.mid_channels3, configs.mid_channels3, configs.kernel_size, configs.stride,
                               configs.padding)
        self.block6 = CNNBlock(configs.mid_channels3, configs.mid_channels2, configs.kernel_size, configs.stride,
                               configs.padding)
        self.block7 = CNNBlock(configs.mid_channels2, configs.mid_channels2, configs.kernel_size, configs.stride,
                               configs.padding)
        self.block8 = CNNBlock(configs.mid_channels2, configs.mid_channels, configs.kernel_size, configs.stride,
                               configs.padding)
        self.block9 = CNNBlock(configs.mid_channels, configs.final_out_channels, configs.kernel_size, configs.stride,
                               configs.padding)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

        # weights_init(self.conv_block1)
        # weights_init(self.conv_block2)
        # weights_init(self.conv_block3)

    def forward(self, x_in):
        x = self.block1(x_in)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat

class MLP(nn.Module):
    def __init__(self, configs):
        super(MLP, self).__init__()
        self.num_layers = configs.mlp_num_layers

        if configs.activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif configs.activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif configs.activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        elif configs.activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif configs.activation.lower() == "leakyrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {configs.activation}")

        self.layers = nn.ModuleList()
        # 添加第一层
        self.layers.append(nn.Linear(configs.input_channels*configs.seq_length, configs.mid_channels4))
        self.layers.append(self.activation)
        self.layers.append(nn.Dropout(configs.dropout))

        for _ in range(self.num_layers - 2):
            self.layers.append(nn.Linear(configs.mid_channels4, configs.mid_channels4))
            self.layers.append(self.activation)
            self.layers.append(nn.Dropout(configs.dropout))

        self.layers.append(nn.Linear(configs.mid_channels4, configs.final_out_channels))

    def forward(self, x):
        """
        前向传播函数。
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, input_channels, seq_length]
        返回:
            Tensor: 输出张量，形状为 [batch_size, final_out_channels]
        """
        x = x.view(x.shape[0], -1) # 最后两个维度合并
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

class DNN(nn.Module):
    def __init__(self, configs):
        super(DNN, self).__init__()
        self.num_layers = configs.DNN_num_layers  # 总层数
        self.input_channels = configs.input_channels
        self.mid_channels4 = configs.mid_channels4
        self.final_out_channels = configs.final_out_channels
        self.dropout = configs.dropout
        self.activation_name = configs.activation.lower()

        # 根据配置选择激活函数
        if self.activation_name == "relu":
            self.activation = nn.ReLU()
        elif self.activation_name == "tanh":
            self.activation = nn.Tanh()
        elif self.activation_name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif self.activation_name == "gelu":
            self.activation = nn.GELU()
        elif self.activation_name == "leakyrelu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")

        # 构建网络层
        self.layers = nn.ModuleList()
        # 添加第一层
        self.layers.append(nn.Linear(configs.input_channels*configs.seq_length, configs.mid_channels4))
        self.layers.append(self.activation)
        self.layers.append(nn.Dropout(self.dropout))

        # 添加中间层
        for _ in range(self.num_layers - 2):
            self.layers.append(nn.Linear(configs.mid_channels4, configs.mid_channels4))
            self.layers.append(self.activation)
            self.layers.append(nn.Dropout(self.dropout))

        # 添加最后一层
        self.layers.append(nn.Linear(configs.mid_channels4, configs.final_out_channels))

    def forward(self, x):
        """
        前向传播函数。
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, input_channels, seq_length]
        返回:
            Tensor: 输出张量，形状为 [batch_size, output_dim]
        """
        x = x.view(x.shape[0], -1) # 最后两个维度合并
        for layer in self.layers:
            x = layer(x)
        return x

class LSTMNet(nn.Module):
    def __init__(self, configs):
        super(LSTMNet, self).__init__()

        self.lstm_num_layers = configs.lstm_num_layers
        self.hidden_dim = configs.mid_channels4

        # 定义 LSTM 层
        self.lstm = nn.LSTM(configs.input_channels, configs.mid_channels4, configs.lstm_num_layers, batch_first=True, dropout=configs.dropout)
        
        # 定义全连接层，将 LSTM 的输出映射到 final_out_channels
        self.fc = nn.Linear(configs.mid_channels4, configs.final_out_channels)


    def forward(self, x):
        """
        前向传播函数。
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, sequence_length, input_dim]
        返回:
            Tensor: 输出张量，形状为 [batch_size, output_dim]
        """
        x = x.permute(0, 2, 1)
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.lstm_num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, x.size(0), self.hidden_dim).to(x.device)
        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出
        out = out[:, -1, :]  # 形状为 [batch_size, hidden_dim]
        out = self.fc(out)

        return out

# class PositionalEncoding(nn.Module):
#     def __init__(self, feature_dim, dropout, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         # 初始化位置编码: [max_len, feature_dim]
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         # 处理 feature_dim 可能是奇数的情况
#         div_term_length = feature_dim // 2  # 计算出偶数维度的数量
#         div_term = torch.exp(
#             torch.arange(0, div_term_length).float() * (-torch.log(torch.tensor(10000.0)) / feature_dim))
#         pe = torch.zeros(max_len, feature_dim)
#         pe[:, 0:2 * div_term_length:2] = torch.sin(position * div_term)  # 偶数列使用 sin
#         pe[:, 1:2 * div_term_length:2] = torch.cos(position * div_term)  # 奇数列使用 cos
#         # 如果 feature_dim 是奇数，保留最后一列为零
#         if feature_dim % 2 == 1:
#             pe[:, -1] = 0
#         pe = pe.unsqueeze(0).transpose(0, 1)  # 变换维度方便应用
#         # 注册位置编码为模型参数，防止在反向传播中被更新
#         self.register_buffer('pe', pe)
#         # 定义一维卷积
#         self.conv1d = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1)
#
#     def forward(self, X):
#         # X 的形状为 [batch_size, channels, seq_length, feature_dim] -> [32, C, 128, 9]
#         batch_size, channels, seq_length, feature_dim = X.shape
#         # 加入位置编码: 提取前 seq_length 个位置编码
#         pos_encoding = self.pe[:seq_length, :].transpose(0,
#                                                          1)  # [seq_length, feature_dim] -> [1, seq_length, feature_dim]
#         # 位置编码和每个通道的特征相加
#         X = X + pos_encoding.unsqueeze(0).unsqueeze(0)  # 扩展位置编码到 [1, 1, seq_length, feature_dim]，与 X 维度匹配
#         # 将维度转换为适合 1D 卷积的格式 [batch_size * channels, feature_dim, seq_length]
#         X = X.view(batch_size * channels, feature_dim, seq_length)
#         # 应用 1D 卷积
#         X = self.conv1d(X)
#         # 转换回 [batch_size, channels, seq_length, feature_dim]
#         X = X.view(batch_size, channels, seq_length, feature_dim)
#         # 再加上 dropout
#         X = self.dropout(X)
#         return X


class PositionalEncoding(nn.Module):
    def __init__(self, feature_dim, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 定义一维卷积层，输入维度是 feature_dim，输出维度可以是任意设定值
        self.conv1d = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, X):
        # 假设输入 X 的形状为 (batch_size, channels, seq_length, feature_dim)
        batch_size, channels, seq_length, feature_dim = X.shape
        # 将 X reshape 为适合一维卷积操作的形状： (batch_size * channels, feature_dim, seq_length)
        X = X.view(batch_size * channels, feature_dim, seq_length)
        # 通过一维卷积进行位置编码学习
        X = self.conv1d(X)
        X = self.relu(X)
        # 将 X 转换回原始形状： (batch_size, channels, seq_length, feature_dim)
        X = X.view(batch_size, channels, seq_length, feature_dim)
        return self.dropout(X)


class TransformerBlock(nn.Module):
    def __init__(self, feature_dim, seq_length, num_heads=1, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention1 = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout)
        self.add_norm1 = self.AddNorm(feature_dim, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.add_norm2 = self.AddNorm(feature_dim, dropout)

    def forward(self, x):
        batch_size, interval, sequence_length, feature_dim = x.shape
        x = x.view(batch_size * interval, sequence_length, feature_dim).transpose(0, 1)

        # Apply the first attention mechanism
        attn_output1, _ = self.attention1(x, x, x)
        y = self.add_norm1(x, attn_output1)

        # Apply the feed-forward network
        ff_output = self.feed_forward(y)
        z = self.add_norm2(y, ff_output)

        z = z.transpose(0, 1).view(batch_size, interval, sequence_length, feature_dim)
        return z

    class AddNorm(nn.Module):
        def __init__(self, normalized_shape, dropout=0.1):
            super(TransformerBlock.AddNorm, self).__init__()
            self.norm = nn.LayerNorm(normalized_shape)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, sublayer_output):
            return self.norm(x + self.dropout(sublayer_output))

# class TransformerBlock(nn.Module):
#     def __init__(self, feature_dim, seq_length, num_heads=1, dropout=0.1):
#         super(TransformerBlock, self).__init__()
#         self.attention1 = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout)
#         self.add_norm1 = self.AddNorm(feature_dim, dropout)
#         self.attention2 = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout)
#         self.add_norm2 = self.AddNorm(feature_dim, dropout)
#
#     def forward(self, x):
#         batch_size, interval, sequence_length, feature_dim = x.shape
#         x = x.view(batch_size * interval, sequence_length, feature_dim).transpose(0, 1)
#         attn_output1, _ = self.attention1(x, x, x)
#         y = self.add_norm1(x, attn_output1)
#         attn_output2, _ = self.attention2(y, y, y)
#         z = self.add_norm2(y, attn_output2)
#         z = z.transpose(0, 1).view(batch_size, interval, sequence_length, feature_dim)
#         return z
#
#     class AddNorm(nn.Module):
#         def __init__(self, normalized_shape, dropout=0.1):
#             super(TransformerBlock.AddNorm, self).__init__()
#             self.norm = nn.LayerNorm(normalized_shape)
#             self.dropout = nn.Dropout(dropout)
#
#         def forward(self, x, sublayer_output):
#             return self.norm(x + self.dropout(sublayer_output))
#
#
# class TransformerBlock(nn.Module):
#     def __init__(self, feature_dim, seq_length, num_heads=1, dropout=0.1):
#         super(TransformerBlock, self).__init__()
#         self.num_heads = num_heads
#         self.temporal_attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout)
#         self.spatial_attention = nn.MultiheadAttention(seq_length, num_heads, dropout=dropout)
#         self.mlp = nn.Linear(2 * feature_dim, feature_dim)  # 定义一个线性层
#         # self.cross_attention = nn.MultiheadAttention(seq_length * feature_dim, num_heads, dropout=dropout)
#         self.cross_attention = nn.MultiheadAttention(seq_length, num_heads, dropout=dropout)
#
#         self.add_norm1 = self.AddNorm(feature_dim, dropout)
#         self.add_norm2 = self.AddNorm(feature_dim, dropout)
#         # self.add_norm2 = self.AddNorm(seq_length, dropout)
#
#     def forward(self, x):
#         # return x.unsqueeze(1)
#         batch_size, interval, sequence_length, feature_dim = x.shape
#         x = x.view(batch_size * interval, sequence_length, feature_dim)
#
#         # self
#         temporal_x = x.transpose(0, 1)  # (sequence_length, batch_size * interval, feature_dim)
#         temporal_output, _ = self.temporal_attention(temporal_x, temporal_x, temporal_x)  # 以每个时间步为token作自注意力
#         temporal_output = temporal_output.transpose(0, 1)  # (batch_size * interval, sequence_length, feature_dim)
#         # print('temporal output shape',temporal_output.shape)
#
#         spatial_x = x.permute(2, 0, 1)
#         spatial_output, _ = self.spatial_attention(spatial_x, spatial_x, spatial_x)  # 以每个通道为token作自注意力
#         spatial_output = spatial_output.permute(1, 2, 0)
#         # print('spatial output shape',spatial_output.shape)
#
#         # concat
#         y_concat = torch.cat((temporal_output, spatial_output),
#                              dim=-1)  # (batch_size * interval, sequence_length, 2 * feature_dim)
#         # mlp
#         # y_self = self.mlp(y_concat)
#         # y_self = temporal_output + spatial_output
#         y_self = spatial_output
#
#         # add&norm
#         y_self_norm = self.add_norm1(x, y_self)
#         # print('y_self_norm',y_self_norm.shape)
#
#         # cross交叉注意力机制
#         num_subwaves = interval
#         # 将输入  按照 interval 切分成 num_subwaves 个子波
#         subwaves = y_self_norm.view(batch_size, num_subwaves, sequence_length, feature_dim)
#         # subwaves = x.view(batch_size, num_subwaves, sequence_length, feature_dim)
#         all_output = []
#         for i in range(num_subwaves):
#             query = subwaves[:, i, :, :].unsqueeze(1)  # (batch_size, 1, sequence_length, feature_dim) 增加一个维度以匹配注意力机制维度
#             query = query.view(batch_size, sequence_length, feature_dim)  # 调整形状 (batch_size, sequence_length, feature_dim)
#             query = query.permute(0, 2, 1)
#             # query.transpose(0, 1)
#
#             key = subwaves[:, torch.arange(num_subwaves) != i, :, :].view(batch_size, num_subwaves - 1, sequence_length, feature_dim) # 其他作为key
#             # key = key.mean(dim=1).unsqueeze(1)  # 平均其他子波的 Key
#
#             key = key.view(batch_size * (num_subwaves - 1), sequence_length, feature_dim)
#             value = key.clone()  # 假设Key和Value相同
#             key = key.permute(0, 2, 1)  # 调整形状 (batch_size * (num_subwaves - 1), feature_dim, sequence_length)
#             value = value.permute(0, 2, 1)
#
#             # 做交叉注意力
#             output, _ = self.cross_attention(query, key, value)
#             all_output.append(output)
#
#         # 将所有子波的输出拼接起来
#         y_cross = torch.cat(all_output, dim=0)  # (batch_size, num_subwaves, sequence_length, feature_dim)
#         y_cross = y_cross.permute(0, 2, 1)
#
#         # add&norm
#         y_cross_norm = self.add_norm2(y_self_norm, y_cross)
#         # y_cross_norm = self.add_norm2(x, y_cross)
#         # print('y_cross_norm  ', y_cross_norm.shape)
#         z = y_cross_norm.unsqueeze(1)
#         return z
#         # return y_self_norm.unsqueeze(1)
#
#     class AddNorm(nn.Module):
#         def __init__(self, normalized_shape, dropout=0.1):
#             super(TransformerBlock.AddNorm, self).__init__()
#             self.norm = nn.LayerNorm(normalized_shape)
#             self.dropout = nn.Dropout(dropout)
#
#         def forward(self, x, sublayer_output):
#             return self.norm(x + self.dropout(sublayer_output))


class TransformerModel(nn.Module):
    def __init__(self, configs):
        super(TransformerModel, self).__init__()
        self.interval = configs.interval
        self.feature_dim = configs.feature_dim
        self.seq_length = configs.seq_length // configs.interval
        self.dropout = configs.dropout
        self.num_layers = configs.num_layers
        self.pos_encoder = PositionalEncoding(self.feature_dim, configs.dropout)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.feature_dim, self.seq_length, 1, configs.dropout) for _ in range(configs.num_layers)]
        )

        self.conv_blocks = nn.ModuleList([
            self.create_conv_block(i, configs) for i in range(configs.num_layers)
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def create_conv_block(self, i, configs):
        # 根据不同的 i 生成对应的卷积块
        if i == 0:
            return nn.Sequential(
                nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                          stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
                nn.BatchNorm1d(configs.mid_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(configs.dropout)
            )
        elif i == 1:
            return nn.Sequential(
                nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
                nn.BatchNorm1d(configs.mid_channels * 2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            )
        elif i == 2:
            return nn.Sequential(
                nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                nn.BatchNorm1d(configs.final_out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
            )

    def split_series(self, data, interval):
        # 获取数据的shape
        batch_size, seq_length, num_features = data.shape
        # 计算截尾后的序列长度，使其可以被整除
        truncated_length = (seq_length // interval) * interval
        # 截尾
        truncated_data = data[:, :truncated_length, :]
        # 分割并堆叠
        sub_series = [truncated_data[:, i::interval, :] for i in range(interval)]
        return torch.stack(sub_series, dim=1)

    def forward(self, x_in):
        x_in = x_in.transpose(1, 2)
        sub_series = self.split_series(x_in, self.interval)
        # X = self.pos_encoder(sub_series * math.sqrt(self.feature_dim)) #96*1*128*9
        X = sub_series #先不要pos
        for transformer in self.transformer_blocks:
            X = transformer(X)
        # for i in range(self.num_layers):
        #     batch_size, interval, seq_length, feature_dim = X.shape
        #     # 获取当前的 transformer_block 和 conv_block
        #     transformer_block = TransformerBlock(feature_dim, seq_length, 1, self.dropout)
        #     conv_block = self.conv_blocks[i]
        #     # 确保 transformer_block 和 conv_block 在与 X 相同的设备上
        #     device = X.device
        #     transformer_block.to(device)
        #     # 通过 transformer_block
        #     X = transformer_block(X)
        #     # 调整维度以适应卷积层的输入
        #     X = X.permute(0, 1, 3, 2)
        #     X = X.contiguous().view(X.size(0) * X.size(1), X.size(2), X.size(3))
        #     # 经过卷积块
        #     X = conv_block(X)
        #     # 恢复维度
        #     X = X.view(batch_size, interval, X.size(1), X.size(2))
        #     X = X.permute(0, 1, 3, 2)
        batch_size, interval, seq_length, feature_dim = X.shape
        X = X.contiguous().view(batch_size, seq_length * interval, feature_dim) #96*18*32
        X = X.permute(0, 2, 1)
        X = self.adaptive_pool(X)
        X = X.squeeze(-1)
        return X

class ItranModel(nn.Module):
    def __init__(self, configs):
        super(ItranModel, self).__init__()
        # self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention  # 是否输出注意力权重
        self.use_norm = configs.use_norm
        # Embedding 在这里做variate-token
        self.enc_embedding = DataEmbedding_inverted(configs.seq_length, configs.d_model, configs.dropout)
        # Encoder-only architecture 多个
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,  # 前馈网络
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)  # 将编码器的输出映射到预测长度。
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forecast(self, x_enc):
        # temporal layer norm
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding inverted
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, None)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules 自注意力、层归一化、前馈网络使用原生transformer
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # # B N E -> B N S -> B S N  映射到预测长度，并将维度inverted回去  从最开始的B L N -> B S N
        # dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates
        #
        # if self.use_norm:
        #     # De-Normalization from Non-stationary Transformer
        #     dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        #     dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return enc_out

    def forward(self, x_in):
        x_in = x_in.transpose(1, 2)
        out = self.adaptive_pool(self.forecast(x_in)).squeeze(-1)
        # out = self.forecast(x_in)
        # out = out.view(out.size(0), -1)
        return out


class CNN_T(nn.Module):
    def __init__(self, configs):
        super(CNN_T, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels_t, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels_t),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels_t, configs.mid_channels_t * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels_t * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels_t * 2, configs.final_out_channels_t, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels_t),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

        # weights_init(self.conv_block1)
        # weights_init(self.conv_block2)
        # weights_init(self.conv_block3)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, 1)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions.to(torch.float64)


# class classifier(nn.Module):
#     def __init__(self, configs):
#         super(classifier, self).__init__()
#         input_dim = 512 * 4
#         hidden_dim1 = 512  # 第一个隐藏层维度
#         hidden_dim2 = 256  # 第二个隐藏层维度
#         hidden_dim3 = 128  # 第三个隐藏层维度
#
#         # 第一个全连接块
#         self.fc_block1 = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim1, bias=False),
#             nn.BatchNorm1d(hidden_dim1),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
#
#         # 第二个全连接块
#         self.fc_block2 = nn.Sequential(
#             nn.Linear(hidden_dim1, hidden_dim2, bias=False),
#             nn.BatchNorm1d(hidden_dim2),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
#
#         # 第三个全连接块
#         self.fc_block3 = nn.Sequential(
#             nn.Linear(hidden_dim2, hidden_dim3, bias=False),
#             nn.BatchNorm1d(hidden_dim3),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
#
#         # 最终分类层
#         self.logits = nn.Linear(hidden_dim3, 1)
#
#     def forward(self, x):
#         x = self.fc_block1(x)
#         x = self.fc_block2(x)
#         x = self.fc_block3(x)
#         predictions = self.logits(x)
#         return predictions.to(torch.float64)




class classifier2(nn.Module):
    def __init__(self, configs):
        super(classifier2, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels * 2, 1)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions.to(torch.float64)

# class classifier2(nn.Module):
#     def __init__(self, configs):
#         super(classifier2, self).__init__()
#         input_dim = 512 * 4 * 2
#         hidden_dim1 = 512  # 第一个隐藏层维度
#         hidden_dim2 = 256  # 第二个隐藏层维度
#         hidden_dim3 = 128  # 第三个隐藏层维度
#
#         # 第一个全连接块
#         self.fc_block1 = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim1, bias=False),
#             nn.BatchNorm1d(hidden_dim1),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
#
#         # 第二个全连接块
#         self.fc_block2 = nn.Sequential(
#             nn.Linear(hidden_dim1, hidden_dim2, bias=False),
#             nn.BatchNorm1d(hidden_dim2),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
#
#         # 第三个全连接块
#         self.fc_block3 = nn.Sequential(
#             nn.Linear(hidden_dim2, hidden_dim3, bias=False),
#             nn.BatchNorm1d(hidden_dim3),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
#
#         # 最终分类层
#         self.logits = nn.Linear(hidden_dim3, 1)
#
#     def forward(self, x):
#         x = self.fc_block1(x)
#         x = self.fc_block2(x)
#         x = self.fc_block3(x)
#         predictions = self.logits(x)
#         return predictions.to(torch.float64)


class classifier_T(nn.Module):
    def __init__(self, configs):
        super(classifier_T, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels_t, configs.num_classes)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions


########## TCN #############################
torch.backends.cudnn.benchmark = True  # might be required to fasten TCN


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):
    def __init__(self, configs):
        super(TCN, self).__init__()

        in_channels0 = configs.input_channels
        out_channels0 = configs.tcn_layers[1]
        kernel_size = configs.tcn_kernel_size
        stride = 1
        dilation0 = 1
        padding0 = (kernel_size - 1) * dilation0

        self.net0 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(out_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
        )

        self.downsample0 = nn.Conv1d(in_channels0, out_channels0, 1) if in_channels0 != out_channels0 else None
        self.relu = nn.ReLU()

        in_channels1 = configs.tcn_layers[0]
        out_channels1 = configs.tcn_layers[1]
        dilation1 = 2
        padding1 = (kernel_size - 1) * dilation1
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
            nn.Conv1d(out_channels1, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
        )
        self.downsample1 = nn.Conv1d(out_channels1, out_channels1, 1) if in_channels1 != out_channels1 else None

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False, padding=padding0,
                      dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),

            nn.Conv1d(out_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding0, dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(out_channels0, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),

            nn.Conv1d(out_channels1, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
        )

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x0 = self.conv_block1(inputs)
        res0 = inputs if self.downsample0 is None else self.downsample0(inputs)
        out_0 = self.relu(x0 + res0)

        x1 = self.conv_block2(out_0)
        res1 = out_0 if self.downsample1 is None else self.downsample1(out_0)
        out_1 = self.relu(x1 + res1)

        out = out_1[:, :, -1]
        return out


######## RESNET ##############################################

class RESNET18(nn.Module):
    def __init__(self, configs):
        layers = [2, 2, 2, 2]
        # block = BasicBlock
        block = BasicBlock1d

        self.inplanes = configs.input_channels
        super(RESNET18, self).__init__()
        self.layer1 = self._make_layer(block, configs.mid_channels, layers[0], stride=configs.stride)
        self.layer2 = self._make_layer(block, configs.mid_channels * 2, layers[1], stride=1)
        self.layer3 = self._make_layer(block, configs.final_out_channels, layers[2], stride=1)
        self.layer4 = self._make_layer(block, configs.final_out_channels, layers[3], stride=1)

        self.avgpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class RESNET34(nn.Module):
    def __init__(self, configs):
        layers = [3, 4, 6, 3]
        block = BasicBlock1d

        self.inplanes = configs.input_channels
        super(RESNET34, self).__init__()
        self.layer1 = self._make_layer(block, configs.mid_channels, layers[0], stride=configs.stride)
        self.layer2 = self._make_layer(block, configs.mid_channels * 2, layers[1], stride=1)
        self.layer3 = self._make_layer(block, configs.final_out_channels, layers[2], stride=1)
        self.layer4 = self._make_layer(block, configs.final_out_channels, layers[3], stride=1)

        self.avgpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


def conv(in_planes, out_planes, stride=1, kernel_size=3):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)


class BasicBlock1d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        # if(isinstance(kernel_size,int)): kernel_size = [kernel_size,kernel_size//2+1]

        self.conv1 = conv(inplanes, planes, stride=stride, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes,kernel_size=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class BasicBlock1d_wang(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,kernel_size=[5,3]):
        super().__init__()

        # if(isinstance(kernel_size,int)): kernel_size = [kernel_size,kernel_size//2+1]

        self.conv1 = conv(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes,kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class RESNET1D_WANG(nn.Module):
    def __init__(self, configs):
        layers = [1,1,1]
        block = BasicBlock1d_wang

        self.input_channels = configs.input_channels
        self.inplanes = configs.mid_channels
        super(RESNET1D_WANG, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(self.input_channels, configs.mid_channels, kernel_size=7, stride=1, padding=3,bias=False),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(inplace=True)
        )

        self.layer2 = self._make_layer(block, configs.mid_channels, layers[0], stride=configs.stride)
        self.layer3 = self._make_layer(block, configs.mid_channels * 2, layers[1], stride=1)
        self.layer4 = self._make_layer(block, configs.final_out_channels, layers[2], stride=1)

        self.avgpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat

##################################################
##########  OTHER NETWORKS  ######################
##################################################

class codats_classifier(nn.Module):
    def __init__(self, configs):
        super(codats_classifier, self).__init__()
        model_output_dim = configs.features_len
        self.hidden_dim = configs.hidden_dim
        self.logits = nn.Sequential(
            nn.Linear(model_output_dim * configs.final_out_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, configs.num_classes))

    def forward(self, x_in):
        predictions = self.logits(x_in)
        return predictions


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, len(configs.scenarios))
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

class Discriminator_t(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self,configs):
        """Init discriminator."""
        super(Discriminator_t, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(128, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

class Discriminator_fea(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator_fea, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels_t, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 1),
            nn.Sigmoid(), #映射到介于0,1之间的值
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

class Adapter(nn.Module):
    """mapping student feature dimension to teacher feature dimension"""

    def __init__(self, configs):
        """Init adaptor."""
        super(Adapter, self).__init__()
        self.layer = nn.Linear(configs.final_out_channels, configs.final_out_channels_t)

    def forward(self, input):
        """Forward the adaptor."""
        out = self.layer(input)
        return out


#### Codes required by DANN ##############
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


#### Codes required by CDAN ##############
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class Discriminator_CDAN(nn.Module):
    """Discriminator model for CDAN ."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator_CDAN, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels * configs.num_classes, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out



