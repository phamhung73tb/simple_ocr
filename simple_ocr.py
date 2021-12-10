#
# You can modify this files
#
import random
import torch
from torchvision import models, transforms
import cv2


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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, len(self.labels))
        self.model.load_state_dict(torch.load('model_resnet.pt'))
        self.model.eval()
        self.model.to(self.device)

    # TODO: implement find label
    def find_label(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            y = self.model(img)
            label = y.detach().cpu().numpy()[0]
        return self.labels[label]
