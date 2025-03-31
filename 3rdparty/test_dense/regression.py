import torch
import torch.nn as nn
import torch.optim as optim
from dense import Network

# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 线性层

    def forward(self, x):
        return self.linear(x)

# 超参数
input_dim = 32  # 输入维度
output_dim = 128  # 输出维度
learning_rate = 0.01
num_epochs = 1000

# 创建模型
model = LinearRegressionModel(input_dim, output_dim)

# 损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 生成随机数据 (假设我们有 1000 个样本)
num_samples = 1000
X = torch.randn(num_samples, input_dim)  # 输入数据
y = torch.randn(num_samples, output_dim)  # 目标数据

# 训练模型
for epoch in range(num_epochs):
    # 前向传播
    predictions = model(X)
    loss = criterion(predictions, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    test_input = torch.randn(1, input_dim)  # 随机测试输入
    predicted_output = model(test_input)
    val_losss = criterion(predicted_output.squeeze(0), y)
    print("Validation Loss:", val_losss.item())
    # print("\nTest Input:")
    # print(test_input)
    # print("\nPredicted Output:")
    # print(predicted_output)