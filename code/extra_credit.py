import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
import math
from typing import Tuple, List
from wire_diagram.wire_diagram_data_loader import WireDiagramDataLoader
import utils
from config import init_logging


class T1WireDiagramDataset(Dataset):

    def __init__(self, dataset_size: int):
        dataset = WireDiagramDataLoader(dataset_size).load_safety_data(
            non_linear_features=False)[0]
        self.x = [elem[0] for elem in dataset]
        self.y = [elem[1] for elem in dataset]
        self.dataset_size = dataset_size

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.dataset_size


class T2WireDiagramDataset(Dataset):

    def __init__(self, dataset_size: int):
        dataset = WireDiagramDataLoader(dataset_size).load_cut_data(
            non_linear_features=False)[0]
        self.x = [elem[0] for elem in dataset]
        self.y = [elem[1] for elem in dataset]
        self.dataset_size = dataset_size

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.dataset_size


def train_logistic(model: any, learning_rate: float, criterion: any, optimizer: any, n_epochs: int, training_dataloader: any, testing_dataloader: any) -> None:
    training_loss, testing_loss = [], []
    for epoch in range(n_epochs):
        # Training
        model.train()
        total_train_loss = 0
        for inputs, labels in training_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        training_loss.append(total_train_loss / len(training_dataloader))

        # Validation
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for inputs, labels in testing_dataloader:
                outputs = model(inputs.float())
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

        testing_loss.append(total_test_loss / len(testing_dataloader))

    plot_loss(training_loss=training_loss, testing_loss=testing_loss,
              data_size=len(training_dataloader), epochs=n_epochs, learning_rate=learning_rate)


def train_softmax(model: any, learning_rate: float, criterion: any, optimizer: any, n_epochs: int, training_dataloader: any, testing_dataloader: any) -> None:
    training_loss, testing_loss = [], []
    for epoch in range(n_epochs):
        # Training
        model.train()
        total_train_loss = 0
        for inputs, labels in training_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            labels = labels.float().squeeze(2)
            # print(outputs, labels)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        training_loss.append(total_train_loss / len(training_dataloader))

        # Validation
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for inputs, labels in testing_dataloader:
                outputs = model(inputs.float())
                labels = labels.float().squeeze(2)
                # print(outputs, labels)
                loss = criterion(outputs, torch.argmax(labels, dim=1))
                total_test_loss += loss.item()

        testing_loss.append(total_test_loss / len(testing_dataloader))


init_logging()


class LogisticRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1600, 100)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(100, 50)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x


class SoftmaxRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1600, 10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10, 4)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


# Task 1
model = LogisticRegression()
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_epochs = 10

training_dataset = T1WireDiagramDataset(2000)
training_dataloader = DataLoader(
    dataset=training_dataset, batch_size=1, shuffle=True)
testing_dataset = T1WireDiagramDataset(2000)
testing_dataloader = DataLoader(
    dataset=testing_dataset, batch_size=10, shuffle=True)
train_logistic(model=model, learning_rate=learning_rate, criterion=criterion, optimizer=optimizer,
               n_epochs=n_epochs, training_dataloader=training_dataloader, testing_dataloader=testing_dataloader)


# Task 2
model = SoftmaxRegression()
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_epochs = 100

training_dataset = T2WireDiagramDataset(2000)
training_dataloader = DataLoader(
    dataset=training_dataset, batch_size=1, shuffle=True)
testing_dataset = T2WireDiagramDataset(2000)
testing_dataloader = DataLoader(
    dataset=testing_dataset, batch_size=1, shuffle=True)
train_softmax(model=model, learning_rate=learning_rate, criterion=criterion, optimizer=optimizer,
              n_epochs=n_epochs, training_dataloader=training_dataloader, testing_dataloader=testing_dataloader)
