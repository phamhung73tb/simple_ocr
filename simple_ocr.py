#
# You can modify this files
#

#
# You can modify this files
#
import random
import torch
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image


class HoadonOCR:
    def __init__(self):
        # Init parameters, load model here
        self.model = None
        self.labels = ['highlands', 'starbucks', 'phuclong', 'others']
        self.class_name = {'highlands': 0, 'starbucks': 1, 'phuclong': 2, 'others': 3}
        self.transform =  transforms.Compose([
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        self.device = torch.device("cpu")
        self.load_model()

    def load_model(self):
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, len(self.labels))
        self.model.load_state_dict(torch.load('model_resnet.pt', map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    # TODO: implement find label
    def find_label(self, img):
        img = Image.fromarray(img).convert('RGB')
        with torch.no_grad():
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(self.device)
            outputs = self.model(img)
            _, preds = torch.max(outputs, 1)
            label = preds.detach().cpu().numpy()[0]
        return self.labels[label]
