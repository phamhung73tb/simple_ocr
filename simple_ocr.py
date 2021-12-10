#
# You can modify this files
#
import random
import torch
from torchvision import models, transforms


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

    def load_model(self):
        self.network = models.resnet18(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = torch.nn.Linear(num_ftrs, len(self.labels))
        self.network.load_state_dict(torch.load('model_resnet.pt'))
        self.network.eval()
        self.network.device()

    # TODO: implement find label
    def find_label(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            y = self.network(img)
            label = y.detach().cpu().numpy()[0]
        return self.class_name[y]
