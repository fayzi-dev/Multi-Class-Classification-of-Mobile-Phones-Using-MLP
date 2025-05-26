import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from torch.testing._internal.common_quantization import AverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0)) #GPU name: GeForce GTX 960M
else:
    print("No GPU available.")


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


from torchmetrics import Accuracy

model = model.to(device)
# Train looooop
num_epochs = 400

loss_train_history = []
acc_train_history = []

loss_valid_history = []
acc_valid_history = []

for epoch in range(num_epochs):
    loss_train = AverageMeter()
    acc_train = Accuracy(task='multiclass', num_classes=4).to(device)
    for i, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        yp = model(x_batch)
        loss = loss_fn(yp, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_train.update(loss.item())
        acc_train(yp, y_batch)

    with torch.no_grad():
        loss_valid = AverageMeter()
        acc_valid = Accuracy(task='multiclass', num_classes=4).to(device)
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            yp = model(x_batch)
            loss = loss_fn(yp, y_batch)
            loss_valid.update(loss.item())
            acc_valid(yp, y_batch)

    loss_train_history.append(loss_train.avg)
    acc_tr = acc_train.compute()
    acc_train_history.append(acc_tr.item())
    loss_valid_history.append(loss_valid.avg)
    acc_v = acc_valid.compute()
    acc_valid_history.append(acc_v.item())

    print(f'Epoch{epoch}')
    print(f'Train Loss:{loss_train.avg}, Acc:{acc_train.compute()}')
    print(f'Valid Loss:{loss_valid.avg}, Acc:{acc_valid.compute()}')
    print()

# plot loss
plt.plot(range(num_epochs), loss_train_history, 'r-', label='Train Loss')
plt.plot(range(num_epochs), loss_valid_history, 'b-', label='Valid Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# plot acc
plt.plot(range(num_epochs), acc_train_history, 'r-', label='Train Accuracy')
plt.plot(range(num_epochs), acc_valid_history, 'b-', label='Valid Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

# save model
torch.save(model, 'model.pt')
# load model
my_model = torch.load('model.pt')
print(my_model)
