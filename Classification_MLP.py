import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.testing._internal.common_quantization import AverageMeter

df = pd.read_csv('train.csv')
# print(df.info())
# print(df.head())
X = df.drop('price_range', axis=1)
# print(X.head())
y = df['price_range']
# print(X.head)


# split dataset
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_valid.shape)  # (1600, 20) (400, 20)
print(y_train.shape, y_valid.shape)  # (1600,) (400,)

x_train = torch.FloatTensor(X_train.values)
y_train = torch.LongTensor(y_train.values)  # convert to integer

x_valid = torch.FloatTensor(X_valid.values)
y_valid = torch.LongTensor(y_valid.values)

# standardization
mu = x_train.mean(dim=0)
std = x_train.std(dim=0)
# print(mu, std)
x_train = (x_train - mu) / std
x_valid = (x_valid - mu) / std

# Dataloader
from torch.utils.data import DataLoader, TensorDataset

train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

print(len(train_loader))  # 25
x_batch, y_batch = next(iter(train_loader))
print(x_batch.shape)  # torch.Size([64, 20])

valid_data = TensorDataset(x_valid, y_valid)
valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

# model
num_features = 20
num_classes = 4

model = nn.Sequential(nn.Linear(num_features, 64),
                      nn.ReLU(),
                      nn.Linear(64, 32),
                      nn.ReLU(),
                      nn.Linear(32, num_classes))

yp = model(x_batch)
# print(yp[:2, :])
xxx = torch.tensor([torch.numel(p) for p in model.parameters()])
# print(xxx)

# loss & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Class AverageMeter
from torch.testing._internal.common_quantization import AverageMeter


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Train looooop
num_epochs = 300

for epoch in range(num_epochs):
    loss_train = AverageMeter()
    for i, (x_batch, y_batch) in enumerate(train_loader):
        yp = model(x_batch)
        loss = loss_fn(yp, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_train.update(loss.item())

    with torch.no_grad():
        loss_valid = AverageMeter()
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            yp = model(x_batch)
            loss = loss_fn(yp, y_batch)
            loss_valid.update(loss.item())

    print(f'Epoch{epoch}')
    print(f'Train Loss:{loss_train.avg}')
    print(f'Valid Loss:{loss_valid.avg}')
    print()
