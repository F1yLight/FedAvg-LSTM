import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])  
        return out

# 参数
input_size = 10
hidden_size = 20
output_size = 1
batch_size = 16
sequence_length = 5

# 实例化模型
model = LSTM(input_size, hidden_size, output_size)

# 生成随机输入数据
x = torch.randn(batch_size, sequence_length, input_size)

# 生成随机目标数据
y = torch.randn(batch_size, output_size)

# 前向传播
output = model(x)

# 计算损失
criterion = nn.MSELoss()
loss = criterion(output, y)

print("Output shape: ", output.shape)
print("Loss: ", loss.item())