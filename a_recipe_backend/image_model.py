import torch
import torchvision.models as models
from torch import nn
from data_images import num_classes
import torch
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from data_images import DataInfo
import os
import json
from IPython.display import Image as Image_Open

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
model = models.resnet18(pretrained=True)

num_classes = 498

model.fc = nn.Linear(model.fc.in_features, num_classes)
try:
    base_dir = os.path.dirname(__file__)  # Path of the script
except NameError:
    base_dir = os.getcwd() 
model_path = os.path.join(base_dir, "../../resnet18_images.pth") 

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
model.eval()
with open("../../encodings.json", "r") as f:
    label_to_index = json.load(f)

index_to_label = {v: k for k, v in label_to_index.items()}

def generate_image_label(img): 
    image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transformed_image = image_transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(transformed_image)  
        predicted_index = torch.argmax(outputs, dim=1).item()  
        predicted_label = index_to_label[predicted_index]  
    print("Predicted Label: ", predicted_label )
    
    
