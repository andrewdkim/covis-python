import json
from os import listdir
import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional
import torch.optim as optim
import time
from torch.utils.data import DataLoader, Subset
import numpy as np


class VGG():
    def __init__(self, saved_checkpoint=None):
    
        vgg = models.vgg16(pretrained=True)
        num_classes = 2  # category a and b

        # freeze layers
        for parameters in vgg.parameters():
            parameters.requires_grad = False
        # remove last layer
        last_layer_size = vgg.classifier[6].in_features

        features = list(vgg.classifier.children())[:-1]
        # features.extend([nn.Linear(last_layer_size, num_classes)])
        print(features)
        vgg.classifier = nn.Sequential(*features)
        self.model = vgg

        # if saved_checkpoint:
        #     self.model.load_state_dict(torch.load(saved_checkpoint))
            
        # #hyperparameters
        # self.num_epochs = 15
        # self.learning_rate = 0.001

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.optimizer = optim.Adam(vgg.classifier.parameters(), lr=self.learning_rate)
        # self.criterion = nn.CrossEntropyLoss()
    def get_feature(self, image):
        self.model.eval()
        logits = self.model(image)
        return logits.numpy().flatten()

    # # validation function
    # def validate(self, test_dataloader):
    #     self.model.eval()
    #     val_running_loss = 0.0
    #     val_running_correct = 0
    #     for i, data in enumerate(test_dataloader):
    #         data, target = data[0].to(self.device), data[1].to(self.device)
    #         output = self.model(data)
    #         loss = self.criterion(output, target)
    #         val_running_loss += loss.item()
    #         _, preds = torch.max(output.data, 1)
    #         val_running_correct += (preds == target).sum().item()
    #     val_loss = val_running_loss/len(test_dataloader.dataset)
    #     val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)
    #     print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')
    #     return val_loss, val_accuracy
    
        

    # def train(self, train_data: DataLoader):
    #     self.model.train()
    #     train_running_loss = 0.0
    #     train_running_correct = 0
    #     for i, data in enumerate(train_data):
    #         data, target = data[0].to(self.device), data[1].to(self.device)
    #         self.optimizer.zero_grad()
    #         output = self.model(data)
    #         loss = self.criterion(output, target)
    #         train_running_loss += loss.item()
    #         _, preds = torch.max(output.data, 1)
    #         train_running_correct += (preds == target).sum().item()
    #         loss.backward()
    #         self.optimizer.step()
    #     train_loss = train_running_loss/len(train_data.dataset)
    #     train_accuracy = 100. * train_running_correct/len(train_data.dataset)
    #     print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    #     return train_loss, train_accuracy
    
    # def run(self, train:DataLoader, test: DataLoader, output_path):
    #     train_loss , train_accuracy = [], []
    #     val_loss , val_accuracy = [], []
    #     for epoch in range(self.num_epochs):
    #         print("Current epoch: {}".format(epoch))
    #         train_epoch_loss, train_epoch_accuracy = self.train(train)
    #         val_epoch_loss, val_epoch_accuracy = self.validate(test)
    #         train_loss.append(train_epoch_loss)
    #         train_accuracy.append(train_epoch_accuracy)
    #         val_loss.append(val_epoch_loss)
    #         val_accuracy.append(val_epoch_accuracy)
    #     torch.save(self.model.state_dict(), output_path + ".pt")
    #     with open(output_path+"_loss_acc.json", 'w') as output_file:
    #         json.dump({
    #             "train_loss": train_loss,
    #             "train_accuracy": train_accuracy,
    #             "val_loss": val_loss, 
    #             "val_accuracy": val_accuracy
    #         }, output_file)
    
    # def predict(self, image):
    #     self.model.eval()
    #     logits = self.model(image[None, ...])
    #     return functional.softmax(logits, dim=1).cpu().data.numpy()
        
    # def predict_class(self, image):
    #     return np.argmax(self.predict(image))
        

    # def create_vectors(self, dataset, output_path):
    #     # freeze layers
    #     for parameters in self.model.parameters():
    #         parameters.requires_grad = False

    #     # remove last layer
    #     features = list(self.model.classifier.children())[:-1]
    #     self.model.classifier = nn.Sequential(*features)

    #     self.model.eval()
    #     res = []

    #     for image, target in dataset:
    #         logits = self.model(image[None, ...])
    #         res.append([target, logits.numpy().flatten().tolist()])
    #     with open(output_path, 'w') as output_file:
    #         json.dump(res, output_file)
    #     return res