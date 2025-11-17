
from torch.fx.experimental.migrate_gradual_types.constraint import F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

from data_util import *
import copy
import math


def time_to_feature(x: torch.Tensor) -> torch.Tensor:
    """Converts a time series tensor to a feature tensor."""
    return x.permute(0, 2, 1)

class TimeBatchNorm2d(nn.BatchNorm1d):
    """A batch normalization layer that normalizes over the last two dimensions of a
    sequence in PyTorch, mimicking Keras behavior.

    This class extends nn.BatchNorm1d to apply batch normalization across time and
    feature dimensions.

    Attributes:
        num_time_steps (int): Number of time steps in the input.
        num_channels (int): Number of channels in the input.
    """

    def __init__(self, normalized_shape: tuple[int, int]):
        """Initializes the TimeBatchNorm2d module.

        Args:
            normalized_shape (tuple[int, int]): A tuple (num_time_steps, num_channels)
                representing the shape of the time and feature dimensions to normalize.
        """
        num_time_steps, num_channels = normalized_shape
        super().__init__(num_channels * num_time_steps)
        self.num_time_steps = num_time_steps
        self.num_channels = num_channels

feature_to_time = time_to_feature

class TimeMixing(nn.Module):
    """Applies a transformation over the time dimension of a sequence.

    This module applies a linear transformation followed by an activation function
    and dropout over the sequence length of the input feature tensor after converting
    feature maps to the time dimension and then back.

    Args:
        input_channels: The number of input channels to the module.
        sequence_length: The length of the sequences to be transformed.
        activation_fn: The activation function to be used after the linear transformation.
        dropout_rate: The dropout probability to be used after the activation function.
    """

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        norm_type: type[nn.Module] = TimeBatchNorm2d,
    ):
        """Initializes the TimeMixing module with the specified parameters."""
        super().__init__()
        self.norm = norm_type((sequence_length, input_channels))
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(sequence_length, sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the time mixing operations on the input tensor.

        Args:
            x: A 3D tensor with shape (N, C, L), where C = channel dimension and
                L = sequence length.

        Returns:
            The normalized output tensor after time mixing transformations.
        """
        x_temp = feature_to_time(
            x
        )  # Convert feature maps to time dimension. Assumes definition elsewhere.
        x_temp = self.activation_fn(self.fc1(x_temp))
        x_temp = self.dropout(x_temp)
        x_res = time_to_feature(x_temp)  # Convert back from time to feature maps.

        return self.norm(x + x_res)  # Apply normalization and combine with original input.

class NBEATSBlock(nn.Module):
    """
    N-BEATS block with backcast for single variable, multivariate input
    """
    def __init__(self, in_channels, seq_len, pred_len,stack_type='normal', hidden_dim=32, theta_dim=4):
        super(NBEATSBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.theta_dim = theta_dim
        self.stack_type = stack_type

        if stack_type=='holiday':
            fc_dim_in = seq_len * in_channels + seq_len + pred_len
            # self.channel_reducer = nn.Linear(in_channels, 1)
            # self.conv1x1 = nn.Conv1d(in_channels, 1, kernel_size=1)
        else:
            fc_dim_in = seq_len * in_channels

        # Fully connected layers for multivariate input
        self.fc_stack = nn.Sequential(
            nn.Linear(fc_dim_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, theta_dim)
        )

        # Backcast for single variable (target variable)
        self.backcast_linear = nn.Linear(theta_dim, seq_len)  # Output [Batch, seq_len]
        # Forecast for single variable
        self.forecast_linear = nn.Linear(theta_dim, pred_len)  # Output [Batch, pred_len]
        self.forecast_linear_gr = nn.Linear(theta_dim, pred_len)
        self.forecast_linear_bt = nn.Linear(theta_dim, pred_len)




    def forward(self, x, holiday):
        # x: [Batch, seq_len, in_channels]
        if self.stack_type=='holiday':
            # x_permuted = x.permute(0, 2, 1)  # [batch, channel, seq]
            # x_reduced = self.conv1x1(x_permuted)  # [batch, 1, seq]
            # x = x_reduced.permute(0, 2, 1)
            weight = torch.nn.Parameter(torch.ones(1, device=x.device))
            weighted_holiday = weight * holiday.squeeze(-1)
            x_flat = x.reshape(x.size(0), -1)
            x_flat = torch.cat((x_flat, weighted_holiday.squeeze(-1)), dim=1)
        else:
            # target_var: [Batch, seq_len] (single target variable for backcast)
            x_flat = x.view(x.size(0), -1)  # [Batch, seq_len * in_channels]
        theta = self.fc_stack(x_flat)  # [Batch, theta_dim]

        # Backcast: reconstruct target variable only
        backcast = self.backcast_linear(theta)  # [Batch, seq_len]
        
        # Forecast: predict future values
        forecast = self.forecast_linear(theta)  # [Batch, pred_len]
        forecast_gr = self.forecast_linear_gr(theta)
        forecast_bt = self.forecast_linear_bt(theta)

        return backcast, forecast, forecast_gr, forecast_bt


class NBEATS(nn.Module):
    """
    N-BEATS for multivariate input, single variable backcast, and univariate output
    """
    def __init__(self, configs):
        super(NBEATS, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.enc_in  # Number of input variables
        self.num_blocks = configs.num_blocks if hasattr(configs, 'num_blocks') else 2
        self.hidden_dim = configs.hidden_dim if hasattr(configs, 'hidden_dim') else 32
        self.theta_dim = configs.theta_dim if hasattr(configs, 'theta_dim') else 4
        self.time_mixing = TimeMixing(
            configs.seq_len,
            configs.enc_in,
            F.relu,
            0.1,
            norm_type=nn.LayerNorm, #nn.LayerNorm
        )
        # Stack of N-BEATS blocks
        self.blocks = nn.ModuleList([
            NBEATSBlock(
                in_channels=self.in_channels,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                hidden_dim=self.hidden_dim,
                theta_dim=self.theta_dim
            ),
            NBEATSBlock(
                in_channels=self.in_channels,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                hidden_dim=self.hidden_dim,
                theta_dim=self.theta_dim,
                stack_type='holiday'
            )
        ])

    def forward(self, x, holiday):
        # holiday_back = holiday[:,:self.seq_len,0]
        # holiday_forward = holiday[:,-self.pred_len:,0]
        # x: [Batch, seq_len, in_channels]
        target_var = x[:, :, 0]  # Assume target variable is the first channel [Batch, seq_len]
        # target_var = target_var * (1 + torch.sigmoid(self.holiday_scale_back) * holiday_back)
        forecast = torch.zeros(x.size(0), self.pred_len, dtype=x.dtype, device=x.device)  # [Batch, pred_len]
        forecast_gr = torch.zeros(x.size(0), self.pred_len, dtype=x.dtype, device=x.device)
        forecast_bt = torch.zeros(x.size(0), self.pred_len, dtype=x.dtype, device=x.device)
        for i, block in enumerate(self.blocks):
            # x = self.time_mixing(x)
            backcast, block_forecast, block_forecast_gr,block_forecast_bt  = block(x, holiday)
            # forecast = forecast * (1 + torch.sigmoid(self.holiday_scale) * holiday_forward)
            forecast = forecast + block_forecast  # Sum forecasts
            forecast_gr = forecast_gr + block_forecast_gr
            forecast_bt = forecast_bt + block_forecast_bt
            target_var = target_var - backcast  # Residual connection for target variable
            # Update input for next block (optional: only update target variable)
            x = x.clone()  # Avoid modifying original input
            x[:, :, 0] = target_var  # Update target variable in input

        return forecast.unsqueeze(-1) ,forecast_gr.unsqueeze(-1),forecast_bt.unsqueeze(-1) # [Batch, pred_len, 1]


class RNNPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, num_layers=1,
                 output_time_steps=8, dropout=0.3, rnn_type='gru',if_I = True):
        """
        RNN预测器（支持GRU和LSTM）

        参数:
            rnn_type: 'gru' 或 'lstm'，指定使用的RNN类型
            其他参数同原GRUPredictor
        """
        super(RNNPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.output_time_steps = output_time_steps
        self.num_layers = num_layers
        self.if_I = if_I
        self.num_targets = output_size
        self.rnn_type = rnn_type.lower()

        # 选择RNN类型
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError("rnn_type must be either 'gru' or 'lstm'")

        # 定义全连接层（与原GRU版本相同）
        self.fc_ili = nn.Linear(hidden_size+output_time_steps*2, output_size * output_time_steps)
        # self.sigma = nn.Linear(hidden_size, output_size * output_time_steps)
        self.interaction = nn.Linear(hidden_size + output_size * output_time_steps*2, hidden_size)
        self.fc_gr = nn.Linear(hidden_size, output_size * output_time_steps)
        self.fc_I_gr = nn.Linear(hidden_size, output_size * output_time_steps)
        self.fc_I = nn.Linear(hidden_size, output_size * output_time_steps)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, holiday):
        # RNN前向传播（处理GRU和LSTM的差异）
        if self.rnn_type == 'gru':
            _, h_n = self.rnn(x)  # GRU返回的h_n是张量
            h_n = h_n[0]  # 取最后一层的隐藏状态
        else:  # LSTM
            _, (h_n, _) = self.rnn(x)  # LSTM返回(h_n, c_n)
            h_n = h_n[-1]  # 取最后一层的隐藏状态

        h_n = self.dropout(h_n)
        holiday = holiday[:, -self.output_time_steps:, :]

        out_beta = self.fc_I(h_n)
        out_beta = out_beta.unsqueeze(-1)
        out_I_gr = self.fc_I_gr(h_n)
        out_I_gr = out_I_gr.unsqueeze(-1)
        out_gr = self.fc_gr(h_n)
        out_gr = out_gr.unsqueeze(-1)


        if self.if_I:
            h_n_with_beta = torch.cat((h_n, out_beta.view(h_n.size(0), -1),out_I_gr.view(h_n.size(0), -1)),dim=-1)
            h_n_interacted = self.interaction(h_n_with_beta)
            out_ili = self.fc_ili(h_n_with_beta)
            # out_sigma = self.sigma(h_n_interacted)
        else:
            out_ili = self.fc_ili(h_n)
            # out_sigma = self.sigma(h_n)
        out_ili = out_ili.unsqueeze(-1)
        # out_sigma = out_sigma.unsqueeze(-1)
        return out_ili, out_gr, out_beta, out_I_gr


class RNNPredictor1(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=1, num_layers=1,
                 output_time_steps=8, dropout=0.3, rnn_type='gru', if_I=True):
        """
        RNN预测器（支持GRU和LSTM）

        参数:
            rnn_type: 'gru' 或 'lstm'，指定使用的RNN类型
            其他参数同原GRUPredictor
        """
        super(RNNPredictor1, self).__init__()
        self.hidden_size = hidden_size
        self.output_time_steps = output_time_steps
        self.num_layers = num_layers
        self.if_I = if_I
        self.num_targets = output_size
        self.rnn_type = rnn_type.lower()

        # 选择RNN类型
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True,
                              )
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError("rnn_type must be either 'gru' or 'lstm'")

        #全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(0.01),
            # nn.Tanh(),
            nn.Linear(128, output_time_steps * 1)  # 输出维度调整为2
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # RNN前向传播（处理GRU和LSTM的差异）
        if self.rnn_type == 'gru':
            _, h_n = self.rnn(x)  # GRU返回的h_n是张量
            h_n = h_n[0]  # 取最后一层的隐藏状态
        else:  # LSTM
            _, (h_n, _) = self.rnn(x)  # LSTM返回(h_n, c_n)
            h_n = h_n[-1]  # 取最后一层的隐藏状态

        h_n = self.dropout(h_n)
        preds = self.fc(h_n).view(-1, self.output_time_steps, 1)

        return preds[:,:,0:1], preds[:,:,0:1], preds[:,:,0:1], preds[:,:,0:1]


def get_weights(x_batch):
    """提取自定义权重"""
    return x_batch[:, -1, :, 2]  # (batch_size, num_features)

def weighted_huber_loss(pred, target, weight):
    """
    pred: 模型输出 (batch_size, seq_len, num_features)
    target: 真实值 (batch_size, seq_len, num_features)
    weight: 自定义权重 (batch_size, num_features)
    """
    # 1. 计算基础Huber损失（逐点计算）
    delta = 0.02  # Huber损失的超参数
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.min(abs_diff, torch.tensor(delta))
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear  # shape: (batch_size, seq_len, num_features)

    # 2. 调整权重形状以匹配loss
    if weight.dim() < loss.dim():
        weight = weight.unsqueeze(1)  # (batch_size, 1, num_features) 适配序列长度

    # 3. 应用权重
    weighted_loss = loss * weight

    # 4. 求均值
    return weighted_loss.mean()

def smoothness_penalty(output, lambda_smooth=0.01):
    diff = output[:, 1:] - output[:, :-1]  # 一阶差分
    second_diff = diff[:, 1:] - diff[:, :-1]  # 二阶差分
    return lambda_smooth * torch.mean(second_diff**2)

def sign_consistency_loss(pred_diff, true_diff):
    # 计算点积
    dot_product = (pred_diff * true_diff).sum(dim=-1)
    # 惩罚负点积（方向相反）
    loss = torch.relu(-dot_product).mean()
    return loss

def quantile_huber_loss(pred, target, quantile=0.3):
    error = pred - target
    huber = torch.where(
        error.abs() <= 1,
        0.5 * error.pow(2),
        error.abs() - 0.5
    )
    mask = (error.detach() > 0).float()  # 高估样本
    return torch.mean(huber * (quantile + mask * (1 - 2 * quantile)))

def maxpool_smooth_torch(data, kernel_size=3, stride=1, padding=1):
    """PyTorch版本的MaxPool平滑（支持GPU和自动微分）"""
    if data.ndim == 2:
        data = data.unsqueeze(1)  # [B,1,T]
    pool = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
    return pool(data)

def moving_average(data, window_size):
    return torch.nn.functional.avg_pool1d(
        data.transpose(1, 2), kernel_size=window_size, stride=1, padding=window_size//2
    ).transpose(1, 2)[:, :data.size(1)-window_size+1, :]

def weighted_mae(pred, target, mask):
    # time_weights = torch.tensor([0.99 ** i for i in range(9)])
    # mask = mask * time_weights  # 融合 mask 和时间权重
    abs_error = torch.abs(pred - target)
    weighted_loss = (abs_error * mask).sum() / (mask.sum() + 1e-6)  # 避免除零
    return weighted_loss


def weighted_mape(pred, target, mask):
    # time_weights = torch.tensor([0.99 ** i for i in range(8)])
    # mask = mask * time_weights  # 融合 mask 和时间权重（可选）

    abs_relative_error = torch.abs((pred - target) / (target + 1e-6))  # 避免除零
    weighted_loss = (abs_relative_error * mask).sum() / (mask.sum() + 1e-6)  # 加权平均
    return weighted_loss * 100  # 转换为百分比
#



def weighted_mae_with_decay(pred, target, mask):
    abs_error = torch.abs(pred - target)  # (batch, seq)

    # 生成线性衰减权重（越近权重越大）
    seq_len = abs_error.shape[1]
    decay_weights = torch.linspace(1.0, 0.1, steps=seq_len, device=abs_error.device)  # 从1.0到0.1线性衰减

    # 合并 mask 和 decay_weights
    weighted_loss = (abs_error * mask * decay_weights).sum() / (mask.sum() + 1e-6)
    return weighted_loss


def weighted_mae_with_exp_decay(pred, target, mask):
    abs_error = torch.abs(pred - target)  # (batch, seq)
    # 生成指数衰减权重（越近权重越大）
    seq_len = abs_error.shape[1]
    decay_weights = torch.exp(-torch.linspace(0, 2, steps=seq_len, device=abs_error.device))  # exp(-x)衰减

    # 合并 mask 和 decay_weights
    weighted_loss = (abs_error * mask * decay_weights).sum() / (mask.sum() + 1e-6)
    return weighted_loss


def masked_nll_loss(mean, sigma, target, mask, lambda_reg=0.01, lambda_target=0.1, alpha=1, lambda_mae=0.1):

    # 负对数似然损失
    error = (target - mean) / (sigma + 1e-6)
    nll_loss = (0.5 * (error ** 2 + torch.log(sigma) + 0.5 * torch.log(torch.tensor(2 * 3.14159))) * mask)
    nll_loss = nll_loss.sum() / (mask.sum() + 1e-6)

    # MAE 损失
    abs_error = torch.abs(mean - target)
    mae_loss = (abs_error * mask).sum() / (mask.sum() + 1e-6)  # 避免除零
    # mae_loss = (torch.abs(target - mean) * mask).sum() / (mask.sum() + 1e-6)

    # 正则化项 1：鼓励方差随时间步递增
    reg_time_loss = 0
    for t in range(1, mean.size(1)):
        reg_time_loss += torch.relu(sigma[:, t - 1] - sigma[:, t]).mean()

    # 正则化项 2：鼓励方差与目标值绝对值成正比
    target_abs = torch.abs(target)
    target_scaled = alpha * target_abs
    reg_target_loss = ((sigma - target_scaled) ** 2 * mask).sum() / (mask.sum() + 1e-6)

    # 总损失
    # total_loss = nll_loss + lambda_mae * mae_loss + lambda_reg * reg_time_loss + lambda_target * reg_target_loss
    total_loss = mae_loss
    return total_loss
def train_trans(model, train_data, val_data, num_epochs=10, batch_size=16, lr=0.001,weight_decay=1e-6, save_path='best_model.pth',plot=True,froze = False):
    train_losses = []
    val_losses = []
    best_model = copy.deepcopy(model)
    train_data = TensorDataset(
        torch.FloatTensor(train_data['X']),
        torch.FloatTensor(train_data['holiday']),
        torch.FloatTensor(train_data['y']),
        torch.FloatTensor(train_data['mask'])
    )

    val_data = TensorDataset(
        torch.FloatTensor(val_data['X']),
        torch.FloatTensor(val_data['holiday']),
        torch.FloatTensor(val_data['y']),
        torch.FloatTensor(val_data['mask'])
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    if froze:
        for name, param in model.named_parameters():
            # print(name)
            if "projector" not in name:
                param.requires_grad = False
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)#, weight_decay=1e-8
    best_val_loss = float('inf')  # 用于跟踪最佳验证损失

    progress_bar = tqdm(range(num_epochs), desc="Training Progress", dynamic_ncols=True, leave=True)
    for epoch in progress_bar:
        model.train()
        train_loss = 0
        for x_batch, holiday, y_batch, mask in train_loader:
            optimizer.zero_grad()
            window_batch = {
                'insample_y':x_batch
            }
            output = model(window_batch)
            total_loss = weighted_mae(output[:,:,0], y_batch[:,:,0], mask[:,:,0])
            total_loss.backward()
            optimizer.step()
            # scheduler.step()
            train_loss += total_loss.item()
        # print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader)}')

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, holiday, y_batch, mask in val_loader:
                window_batch = {
                    'insample_y':x_batch
                }
                output = model(window_batch)
                total_loss = weighted_mae(output[:,:,0], y_batch[:,:,0], mask[:,:,0])
                val_loss += total_loss.item()
                # val_loss += huber_loss(output, y_batch[:, :, 0]).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 只在验证损失更好时保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)  # 深度复制当前最佳模型状态
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': avg_val_loss,
            # }, save_path)
            print(f'New best model saved with val loss: {best_val_loss:.4f}')

        progress_bar.set_postfix({
            "Epoch": f"{epoch + 1}/{num_epochs}",
            "Train Loss": f"{avg_train_loss:.4f}",
            "Val Loss": f"{avg_val_loss:.4f}"
        })
    if plot == True:
        # 绘制训练和验证损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    # best_model = copy.deepcopy(model)
    # # 将模型恢复到最佳状态
    model.load_state_dict(best_model.state_dict())
    return model


def train_tcn(model, train_data, val_data, num_epochs=10, batch_size=16, lr=0.001, weight_decay=1e-7, save_path='best_model.pth',plot=True,froze = False):
    train_losses = []
    val_losses = []
    best_model = copy.deepcopy(model)
    train_data = TensorDataset(
        torch.FloatTensor(train_data['X']),
        torch.FloatTensor(train_data['holiday']),
        torch.FloatTensor(train_data['y']),
        torch.FloatTensor(train_data['mask'])
    )

    val_data = TensorDataset(
        torch.FloatTensor(val_data['X']),
        torch.FloatTensor(val_data['holiday']),
        torch.FloatTensor(val_data['y']),
        torch.FloatTensor(val_data['mask'])
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    if froze:
        for name, param in model.named_parameters():
            if "mlp_decoder" not in name:
                param.requires_grad = False
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)#,, weight_decay=1e-7
    best_val_loss = float('inf')  # 用于跟踪最佳验证损失

    progress_bar = tqdm(range(num_epochs), desc="Training Progress", dynamic_ncols=True, leave=True)
    for epoch in progress_bar:
        model.train()
        train_loss = 0
        for x_batch, holiday, y_batch, mask in train_loader:
            optimizer.zero_grad()
            window_batch = {
                'insample_y': x_batch[:, :, 0:1],
                'hist_exog': x_batch[:, :, 1:],
                'futr_exog': x_batch[:, :, 0:1],
                'stat_exog': x_batch[:, :, 0:1],
                'insample_mask': torch.ones_like(x_batch[:, :, 0:1])
            }
            output = model(window_batch)
            total_loss = weighted_mae(output[:,:,0], y_batch[:,:,0], mask[:,:,0])
            total_loss.backward()
            optimizer.step()
            # scheduler.step()
            train_loss += total_loss.item()
        # print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader)}')

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, holiday, y_batch, mask in val_loader:
                window_batch = {
                    'insample_y': x_batch[:, :, 0:1],
                    'hist_exog': x_batch[:, :, 1:],
                    'futr_exog': x_batch[:, :, 0:1],
                    'stat_exog': x_batch[:, :, 0:1],
                    'insample_mask': torch.ones_like(x_batch[:, :, 0:1])
                }
                output = model(window_batch)
                total_loss = weighted_mae(output[:,:,0], y_batch[:,:,0], mask[:,:,0])
                val_loss += total_loss.item()
                # val_loss += huber_loss(output, y_batch[:, :, 0]).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 只在验证损失更好时保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)  # 深度复制当前最佳模型状态
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': avg_val_loss,
            # }, save_path)
            print(f'New best model saved with val loss: {best_val_loss:.4f}')

        progress_bar.set_postfix({
            "Epoch": f"{epoch + 1}/{num_epochs}",
            "Train Loss": f"{avg_train_loss:.4f}",
            "Val Loss": f"{avg_val_loss:.4f}"
        })
    if plot == True:
        # 绘制训练和验证损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    # best_model = copy.deepcopy(model)
    # # 将模型恢复到最佳状态
    model.load_state_dict(best_model.state_dict())
    return model

def train(model, train_data, val_data, num_epochs=10, batch_size=16, lr=0.001,weight_decay=1e-6, save_path='best_model.pth',plot=True,froze=False):
    train_losses = []
    val_losses = []
    best_model = copy.deepcopy(model)
    train_data = TensorDataset(
        torch.FloatTensor(train_data['X']),
        torch.FloatTensor(train_data['holiday']),
        torch.FloatTensor(train_data['y']),
        torch.FloatTensor(train_data['mask'])
    )

    val_data = TensorDataset(
        torch.FloatTensor(val_data['X']),
        torch.FloatTensor(val_data['holiday']),
        torch.FloatTensor(val_data['y']),
        torch.FloatTensor(val_data['mask'])
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')  # 用于跟踪最佳验证损失

    for name, param in model.named_parameters():
        if froze:
            if 'rnn.weight' in name.lower():
                param.requires_grad = False
                # print(f"冻结层: {name}")
    progress_bar = tqdm(range(num_epochs), desc="Training Progress", dynamic_ncols=True, leave=True)

    for epoch in progress_bar:
        model.train()
        train_loss = 0
        for x_batch, holiday, y_batch, mask in train_loader:
            optimizer.zero_grad()
            output_ili, output_gr,output_bt,output_I_gr = model(x_batch)
            huber_loss_value = weighted_mae(output_ili[:, :, 0], y_batch[:, :, 0],mask[:, :, 0])
            # huber_loss_value = masked_nll_loss(output_ili[:, :, 0],out_sigma[:,:,0], y_batch[:, :, 0], mask[:, :, 0])
            huber_loss_value1 = weighted_mae(output_bt[:, :, 0], y_batch[:, :, 2],mask[:, :, 0])
            huber_loss_value2 = weighted_mae(output_gr[:, :, 0],  y_batch[:, :, 1],mask[:, :, 0])
            huber_loss_value3 = weighted_mae(output_I_gr[:, :, 0], y_batch[:, :, 3], mask[:, :, 0])
            total_loss = huber_loss_value #+ 0.1*huber_loss_value1 + 0.1*huber_loss_value2 + 0.1*huber_loss_value3

            total_loss.backward()
            optimizer.step()
            # scheduler.step()
            train_loss += total_loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, holiday, y_batch, mask in val_loader:
                output,output_gr,output_bt, output_I_gr = model(x_batch)
                huber_loss_value = weighted_mae(output[:, :, 0], y_batch[:, :, 0],mask[:, :, 0])
                # diff_loss = mae_loss(output_gr[:, :, 0], y_batch[:, :, 1])
                total_loss = huber_loss_value    #0.1*diff_loss
                val_loss += total_loss.item()
                # val_loss += huber_loss(output, y_batch[:, :, 0]).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        progress_bar.set_postfix({
            "Epoch": f"{epoch + 1}/{num_epochs}",
            "Train Loss": f"{avg_train_loss:.4f}",
            "Val Loss": f"{avg_val_loss:.4f}"
        })

        # 只在验证损失更好时保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)  # 深度复制当前最佳模型状态
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': avg_val_loss,
            # }, save_path)
            print(f'New best model saved with val loss: {best_val_loss:.4f}')
    if plot == True:
        # 绘制训练和验证损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    # best_model = copy.deepcopy(model)
    # # 将模型恢复到最佳状态
    model.load_state_dict(best_model.state_dict())
    return model


def get_val_predictions(model, val_data, scaler, if_res=False,shuffle_feature=False, feature_idx=None,timestep_idx=None,seed=42):
    model.eval()
    scaler_target = scaler['y']
    X_val = val_data['X'].copy()
    if shuffle_feature:
        if feature_idx is None or feature_idx >= X_val.shape[-1]:
            print(X_val.shape[-1])
            raise ValueError("feature_idx must be specified and within the valid range of features.")
        target_data = X_val[:, :, feature_idx].flatten() # 形状: (n_samples * n_timesteps,)
        rng = np.random.default_rng(seed)
        shuffled_data = rng.permutation(target_data)
        X_val[:, :, feature_idx] = shuffled_data.reshape(X_val.shape[0], X_val.shape[1])

        # if feature_idx is None or feature_idx >= X_val.shape[-1]:
        #     print(X_val.shape[-1])
        #     raise ValueError("feature_idx must be specified and within the valid range of features.")
        # if timestep_idx is None or timestep_idx >= X_val.shape[1]:
        #     raise ValueError("timestep_idx must be specified and within the valid range of timesteps.")
        # # 只打乱指定特征和时间步的样本间数据
        # indices = np.random.permutation(X_val.shape[0])  # 打乱样本索引
        # X_val[:, timestep_idx, feature_idx] = X_val[indices, timestep_idx, feature_idx]  # 按样本索引重新排列
    val_data = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(val_data['holiday']),
        torch.FloatTensor(val_data['y']),
        torch.FloatTensor(val_data['mask'])
    )
    val_loader = DataLoader(val_data, batch_size=32)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, holiday, y_batch,_ in val_loader:
            horzion = y_batch.size(1)
            # x_batch[:, :, 0] = maxpool_smooth_torch(x_batch[:, :, 0]).squeeze(1)  # 直接操作Tensor
            preds_scaled, preds_gr ,output_bt, output_I_gr= model(x_batch)
            pred1 = preds_scaled.clone()
            pred = pred1

            # zero_holiday = torch.zeros_like(future_holiday)
            # preds_scaled,preds_gr = model(x_batch)
            preds_scaled = torch.stack([pred, preds_gr ,output_bt, output_I_gr], dim=-1)  # 形状 (A, B, 2)
            original_shape = preds_scaled.shape
            preds_2d = preds_scaled.reshape(-1, original_shape[-1])
            preds = scaler_target.inverse_transform(preds_2d)
            preds = preds.reshape(original_shape[0], original_shape[1], -1)
            #
            original_shape = y_batch.shape  # 保存原始形状
            y_2d = y_batch.reshape(-1, original_shape[-1])  # -> [batch_size*seq_len, num_features]
            targets = scaler_target.inverse_transform(y_2d)
            targets = targets.reshape(original_shape)  # -> [batch_size, seq_len, num_features]

            # preds = scaler_target.inverse_transform(preds_scaled[:, :, 0].numpy())
            # targets = scaler_target.inverse_transform(y_batch[:, :, 0].numpy())

            all_preds.append(preds[:,:,0])
            all_targets.append(targets[:,:,0])

    # 合并所有batch的结果
    y_val_pred = np.concatenate(all_preds, axis=0)
    y_val_true = np.concatenate(all_targets, axis=0)

    return y_val_true, y_val_pred



def get_val_predictions_trans(model, val_data, scaler, if_res=False,shuffle_feature=False, feature_idx=None,timestep_idx=None,seed=42):
    model.eval()
    scaler_target = scaler['y']
    X_val = val_data['X'].copy()
    if shuffle_feature:
        if feature_idx is None or feature_idx >= X_val.shape[-1]:
            print(X_val.shape[-1])
            raise ValueError("feature_idx must be specified and within the valid range of features.")
        target_data = X_val[:, :, feature_idx].flatten()  # 形状: (n_samples * n_timesteps,)
        rng = np.random.default_rng(seed)
        shuffled_data = rng.permutation(target_data)
        X_val[:, :, feature_idx] = shuffled_data.reshape(X_val.shape[0], X_val.shape[1])


    val_data = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(val_data['holiday']),
        torch.FloatTensor(val_data['y']),
        torch.FloatTensor(val_data['mask'])
    )
    val_loader = DataLoader(val_data, batch_size=32)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, holiday, y_batch,_ in val_loader:
            window_batch = {
                'insample_y': x_batch
            }
            preds_scaled = model(window_batch)
            pred1 = preds_scaled.clone()
            pred = pred1

            # zero_holiday = torch.zeros_like(future_holiday)
            # preds_scaled,preds_gr = model(x_batch)
            preds_scaled = torch.stack([pred, pred ,pred, pred], dim=-1)  # 形状 (A, B, 2)
            original_shape = preds_scaled.shape
            preds_2d = preds_scaled.reshape(-1, original_shape[-1])
            preds = scaler_target.inverse_transform(preds_2d)
            preds = preds.reshape(original_shape[0], original_shape[1], -1)
            #
            original_shape = y_batch.shape  # 保存原始形状
            y_2d = y_batch.reshape(-1, original_shape[-1])  # -> [batch_size*seq_len, num_features]
            targets = scaler_target.inverse_transform(y_2d)
            targets = targets.reshape(original_shape)  # -> [batch_size, seq_len, num_features]

            # preds = scaler_target.inverse_transform(preds_scaled[:, :, 0].numpy())
            # targets = scaler_target.inverse_transform(y_batch[:, :, 0].numpy())

            all_preds.append(preds[:,:,0])
            all_targets.append(targets[:,:,0])

    # 合并所有batch的结果
    y_val_pred = np.concatenate(all_preds, axis=0)
    y_val_true = np.concatenate(all_targets, axis=0)

    return y_val_true, y_val_pred


def get_val_predictions_itrans(model, val_data, scaler, if_res=False,shuffle_feature=False, feature_idx=None,timestep_idx=None,seed=42):
    model.eval()
    scaler_target = scaler['y']
    X_val = val_data['X'].copy()
    if shuffle_feature:
        if feature_idx is None or feature_idx >= X_val.shape[-1]:
            print(X_val.shape[-1])
            raise ValueError("feature_idx must be specified and within the valid range of features.")
        target_data = X_val[:, :, feature_idx].flatten()  # 形状: (n_samples * n_timesteps,)
        rng = np.random.default_rng(seed)
        shuffled_data = rng.permutation(target_data)
        X_val[:, :, feature_idx] = shuffled_data.reshape(X_val.shape[0], X_val.shape[1])
        if feature_idx == 5:
            X_val = replace_zero_with_one(X_val, feature_idx, replace_ratio=0.5)
    val_data = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(val_data['holiday']),
        torch.FloatTensor(val_data['y']),
        torch.FloatTensor(val_data['mask'])
    )

    val_loader = DataLoader(val_data, batch_size=32)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, holiday, y_batch,_ in val_loader:
            window_batch = {
                    'insample_y':x_batch
                }
            output = model(window_batch)
            pred1 = output[:,:,:1].clone()
            pred = pred1

            # zero_holiday = torch.zeros_like(future_holiday)
            # preds_scaled,preds_gr = model(x_batch)
            preds_scaled = torch.stack([pred, pred ,pred, pred], dim=-1)  # 形状 (A, B, 2)
            original_shape = preds_scaled.shape
            preds_2d = preds_scaled.reshape(-1, original_shape[-1])
            preds = scaler_target.inverse_transform(preds_2d)
            preds = preds.reshape(original_shape[0], original_shape[1], -1)
            #
            original_shape = y_batch.shape  # 保存原始形状
            y_2d = y_batch.reshape(-1, original_shape[-1])  # -> [batch_size*seq_len, num_features]
            targets = scaler_target.inverse_transform(y_2d)
            targets = targets.reshape(original_shape)  # -> [batch_size, seq_len, num_features]

            # preds = scaler_target.inverse_transform(preds_scaled[:, :, 0].numpy())
            # targets = scaler_target.inverse_transform(y_batch[:, :, 0].numpy())

            all_preds.append(preds[:,:,0])
            all_targets.append(targets[:,:,0])

    # 合并所有batch的结果
    y_val_pred = np.concatenate(all_preds, axis=0)
    y_val_true = np.concatenate(all_targets, axis=0)

    return y_val_true, y_val_pred

def replace_zero_with_one(X_val, feature_idx, replace_ratio=0.5):
    # 复制数据以避免修改原始数组
    X_modified = X_val.copy()
    # 获取 batch 维度大小
    batch_size = X_val.shape[0]
    num_replace = int(batch_size * replace_ratio)  # 要替换的样本数量
    replace_indices = np.random.choice(batch_size, size=num_replace, replace=False)

    # 将选中的样本在指定特征列替换为全0.13237
    X_modified[replace_indices, :, feature_idx] = 0.13237
    return X_modified

def get_val_predictions_tcn(model, val_data, scaler, if_res=False,shuffle_feature=False, feature_idx=None,timestep_idx=None,seed=42):
    model.eval()
    scaler_target = scaler['y']
    X_val = val_data['X'].copy()
    if shuffle_feature:
        if feature_idx is None or feature_idx >= X_val.shape[-1]:
            print(X_val.shape[-1])
            raise ValueError("feature_idx must be specified and within the valid range of features.")
        target_data = X_val[:, :, feature_idx].flatten()  # 形状: (n_samples * n_timesteps,)
        rng = np.random.default_rng(seed)
        shuffled_data = rng.permutation(target_data)
        X_val[:, :, feature_idx] = shuffled_data.reshape(X_val.shape[0], X_val.shape[1])

    val_data = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(val_data['holiday']),
        torch.FloatTensor(val_data['y']),
        torch.FloatTensor(val_data['mask'])
    )
    val_loader = DataLoader(val_data, batch_size=32)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, holiday, y_batch,_ in val_loader:
            window_batch = {
                'insample_y': x_batch[:, :, 0:1],
                'hist_exog': x_batch[:, :, 1:],
                'futr_exog': x_batch[:, :, 0:1],
                'stat_exog': x_batch[:, :, 0:1],
                'insample_mask': torch.ones_like(x_batch[:, :, 0:1])
            }
            preds_scaled = model(window_batch)
            pred1 = preds_scaled.clone()
            pred = pred1

            # zero_holiday = torch.zeros_like(future_holiday)
            # preds_scaled,preds_gr = model(x_batch)
            preds_scaled = torch.stack([pred, pred ,pred, pred], dim=-1)  # 形状 (A, B, 2)
            original_shape = preds_scaled.shape
            preds_2d = preds_scaled.reshape(-1, original_shape[-1])
            preds = scaler_target.inverse_transform(preds_2d)
            preds = preds.reshape(original_shape[0], original_shape[1], -1)
            #
            original_shape = y_batch.shape  # 保存原始形状
            y_2d = y_batch.reshape(-1, original_shape[-1])  # -> [batch_size*seq_len, num_features]
            targets = scaler_target.inverse_transform(y_2d)
            targets = targets.reshape(original_shape)  # -> [batch_size, seq_len, num_features]

            # preds = scaler_target.inverse_transform(preds_scaled[:, :, 0].numpy())
            # targets = scaler_target.inverse_transform(y_batch[:, :, 0].numpy())

            all_preds.append(preds[:,:,0])
            all_targets.append(targets[:,:,0])

    # 合并所有batch的结果
    y_val_pred = np.concatenate(all_preds, axis=0)
    y_val_true = np.concatenate(all_targets, axis=0)

    return y_val_true, y_val_pred






#
# class Config:
#     seq_len = 12  # Input sequence length
#     pred_len = 8  # Prediction length
#     enc_in = 5    # 5 input variables
#     num_blocks = 2
#     hidden_dim = 32
#     theta_dim = 16
#
# configs = Config()
# model = NBEATS(configs)
# x = torch.randn(32, 12, 5)  # [Batch=32, seq_len=48, channels=5]
# output = model(x)  # [Batch=32, pred_len=24, channels=1]
# print(output)  # torch.Size([32, 24, 1])