from torch.utils.data import DataLoader,Dataset
from torch import nn
import torch
import pandas as pd
class FashionData(Dataset):
    def __init__(self,path_of_folder):
        path_to_features=path_of_folder+"/fashion_train.csv"
        path_to_labels=path_of_folder+"/fashion_train_labels.csv"
        self.X=pd.read_csv(path_to_features).values
        self.y=pd.read_csv(path_to_labels)['0'].values    
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        x=self.X[idx]
        y=self.y[idx]
        batch={'X':x,'y':y}
        return batch
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(784,30)
        self.layer2=nn.Linear(30,30)
        self.layer3=nn.Linear(30,10)
    def forward(self,X):
        X=self.layer1(X)
        X=torch.sigmoid(X)
        X=self.layer2(X)
        X=torch.sigmoid(X)
        X=self.layer3(X)
        X=torch.softmax(X,axis=1)
        return X