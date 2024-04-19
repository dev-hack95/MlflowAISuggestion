import os
import sys
import yaml
import torch
import warnings
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainTestSplit:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def NumpyToTensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x.values).type(torch.float)
        y = torch.from_numpy(self.y.values).type(torch.float)
        return x, y
        
    def SplitData(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_tensor, y_tensor = self.NumpyToTensor()
        x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.8)
        return x_train, x_test, y_train, y_test
    

class ModelV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=8, out_features=128)
        self.layer_2 = nn.Linear(in_features=128, out_features=256)
        self.layer_3 = nn.Linear(in_features=256, out_features=128)
        self.layer_4 = nn.Linear(in_features=128, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        a = self.layer_1(x)
        a = self.relu(a)
        a = self.layer_2(a)
        a = self.relu(a)
        a = self.layer_3(a)
        a = self.relu(a)
        a = self.layer_4(a)
        return a
    
class TrainModelV2:
    def __init__(self, model, x_train, y_train, x_test, y_test, epochs=200, lr=0.01):
        self.epochs = epochs
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_loss = list()
        self.test_loss = list()
        self.epoch_count = list()
        self.loss_fn = nn.L1Loss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        
    def Plot(self):
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.title("Train and Test loss Curves")
        plt.plot(self.epoch_count, self.train_loss, label="Training Loss")
        plt.plot(self.epoch_count, self.test_loss, label="Testing Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

    def Train(self):
        torch.manual_seed(42)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            y_pred = self.model(self.x_train)
            target = self.y_train.view_as(y_pred)
            loss = self.loss_fn(y_pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 50 == 0:
                self.model.eval()
                with torch.inference_mode():
                    self.epoch_count.append(epoch)
                    test_pred = self.model(self.x_test)
                    target_test = self.y_test.view_as(test_pred)
                    test_loss = self.loss_fn(test_pred, target_test)
                    self.train_loss.append(loss.item())
                    self.test_loss.append(test_loss.item())
                    print(f"EpocH: {epoch} | loss: {loss.item()} | test_loss: {test_loss.item()}")


#if __name__ == '__main__':
    #data_1 = TrainTestSplit(x, y)
    #x_train, x_test, y_train, y_test = data_1.SplitData()
    #model_1 = ModelV1().to(device)
    #x_train_device = torch.tensor(x_train, dtype=torch.float, device=device)
    #y_train_device = torch.tensor(y_train, dtype=torch.long, device=device)
    #x_test_device = torch.tensor(x_test, dtype=torch.float, device=device)
    #y_test_device = torch.tensor(y_test, dtype=torch.long, device=device)
    #train_1 = TrainModelV2(model=model_1, x_train=x_train_device, y_train=y_train_device, x_test=x_test_device, y_test=y_test_device, epochs=1500, lr=1e-3)
