
import numpy as np
import torch.nn as nn
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
from torch.utils.data import Dataset, DataLoader


class BertForRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
#         self.dropout = nn.Dropout(p=0.75)
        self.classifier = nn.Linear(768, 1024)
#         self.batchn = torch.nn.BatchNorm1d(1000)
        self.relu = torch.nn.LeakyReLU()
        self.classifier2 = nn.Linear(1024, 1024)
#         self.batchn2 = torch.nn.BatchNorm1d(100)
        self.relu2 = torch.nn.LeakyReLU()
        self.classifier3 = nn.Linear(1024, 256)
        self.relu3 = torch.nn.LeakyReLU()
        self.classifier4 = nn.Linear(256, 64)
        self.relu4 = torch.nn.LeakyReLU()
        self.classifier5 = nn.Linear(64, 1)


    def forward(self, tokens, token_type_ids=None):
        x = self.bert(tokens)
#         print(torch.mean(x[0], dim=1).shape)
#         x = self.dropout(torch.mean(x[0], dim=1))
        x = self.classifier(torch.mean(x[0], dim=1))
#         x = self.batchn(x)
        x = self.relu(x)
        x = self.classifier2(x)
#         x = self.batchn2(x)
        x = self.relu2(x)
        x = self.classifier3(x)
        x = self.relu3(x)
        x = self.classifier4(x)
        x = self.relu4(x)
        x = self.classifier5(x)
        return x

    def train_mode(self,):
        self.bert.eval()
        self.classifier.train()

    def train_mode2(self,):
        self.bert.train()
        self.classifier.train()

    def eval_mode(self,):
        self.bert.eval()
        self.classifier.eval()
