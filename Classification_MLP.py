import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

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

