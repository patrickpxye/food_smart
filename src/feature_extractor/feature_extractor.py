from torchvision import models as models

import torch
import torch.nn as nn
import torch.optim as optim

from deep_learning.engine import train, validate
from torch.utils.data import DataLoader


class FeatureExtractor:
    def __init__(self, out_features=None):
        self.out_features = out_features

    def fixed_extractor(self):
        resnet50_model = models.resnet50(progress=True, pretrained=True)
        # to freeze the hidden layers
        for param in resnet50_model.parameters():
            param.requires_grad = False

        resnet50_model.fc = nn.Identity()
        return resnet50_model

    def trainable_extractor(self):
        resnet50_model = models.resnet50(progress=True, pretrained=True)
        # to freeze the hidden layers
        for param in resnet50_model.parameters():
            param.requires_grad = False

        # make the classification layer learnable
        resnet50_model.fc = nn.Linear(2048, self.out_features)
        return resnet50_model

    def train_extractor(self, train_data, valid_data, model_file, epochs=20, lr=0.0001, batch_size=32):
        # initialize the computation device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # initialize the model
        model = self.trainable_extractor().to(device)

        # learning parameters
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # train data loader
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        )
        # validation data loader
        valid_loader = DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False
        )

        # start the training and validation
        train_loss = []
        valid_loss = []
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} of {epochs}")
            train_epoch_loss = train(
                model, train_loader, optimizer, criterion, train_data, device
            )
            valid_epoch_loss = validate(
                model, valid_loader, criterion, valid_data, device
            )
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f'Val Loss: {valid_epoch_loss:.4f}')

        # save the trained model to disk
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, model_file)

    def load_extractor(self, model_file):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.trainable_extractor().to(device)
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
