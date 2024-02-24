import os
import sys
from glob import glob
import matplotlib.pyplot as plt
from glob import glob
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from src.utils import read_yaml_file
from src.logger import CustomException, logger
from src.config.configuration import ModelTrainingConfig

from pathlib import Path


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform=None):
        self.data = datasets.ImageFolder(root=folder_path, transform=transform)
        self.classes = self.data.classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label
    



class ModelTrainingPytorch:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.train_path = f"{self.config.root_dir}/Train"
        self.test_path = f"{self.config.root_dir}/Test"

        self.model_params = read_yaml_file(Path('params.yaml'))
        self.model_config = self.model_params.MODEL_CONFIG

    def train(self):
        try:
            
            folders = glob(f'{self.train_path}/*')
            num_classes = len(folders)
            ## initialize the model

            model = resnet50(pretrained=True)
            for params in model.parameters():
                params.requires_grad = False

            ## Fully connected layers
            model.fc = nn.Linear(model.fc.in_features, num_classes)

            transform = transforms.Compose([
                transforms.Resize(self.model_config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            train_dataset = CustomDataset(folder_path = self.train_path, transform=transform)
            test_dataset = CustomDataset(folder_path = self.test_path, transform=transform)

            train_loader = DataLoader(train_dataset, batch_size=self.model_config.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.model_config.batch_size, shuffle=True)

            cost_fun = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.fc.parameters())

            NUM_OF_EPOCHS = self.model_config.epochs
            train_losses = []
            test_losses = []
            train_accuracies = []
            test_accuracies = []


            for epoch in range(NUM_OF_EPOCHS):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for images, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = cost_fun(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                
                train_loss = running_loss / len(train_loader)
                train_accuracy = 100. * correct / total

                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)


                ## validation
                model.eval()
                correct = 0
                total = 0
                test_loss = 0

                with torch.no_grad():
                    for images, labels in test_loader:
                        outputs = model(images)
                        loss = cost_fun(outputs, labels)
                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()


                test_loss /= len(test_loader)
                test_accuracy = 100. * correct / total
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)


                logger.info(f'Epoch [{epoch + 1}/{NUM_OF_EPOCHS}], '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Train Accuracy: {train_accuracy:.2f}%, '
                    f'Test Accuracy: {test_accuracy:.2f}%')
                

            plt.figure(figsize=(10, 5))

            # Plotting Loss
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()

            # Plotting Accuracy
            plt.subplot(1, 2, 2)
            plt.plot(train_accuracies, label='Train Accuracy')
            plt.plot(test_accuracies, label='Test Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Curve')
            plt.legend()

            # Save plots
            plt.savefig(f'{self.config.model_dir}/Loss_Accuracy_Curves.png')

            # Save the trained model
            torch.save(model.state_dict(), f'{self.config.model_dir}/model_resnet50.pth')




        except Exception as error:
            logger.error(CustomException(error, sys))


if __name__ == '__main__':
    obj = ModelTrainingPytorch()
    obj.train()
