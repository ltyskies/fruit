import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

n_samples = 100
data = torch.rand(n_samples, 2)
labels = (data[:, 0]**2 + data[:, 1]**2 < 1).float().unsqueeze(1)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = SimpleNN()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    outputs = model(data)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

