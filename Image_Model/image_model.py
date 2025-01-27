import torch
import torchvision.models as models
from torch import nn
from data_images import num_classes

model = models.resnet18(pretrained=True)  


num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes) 


for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

print(model)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
