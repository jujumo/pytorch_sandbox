import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision import models
import torchvision.datasets as datasets
from PIL import Image
from util import showtensor
import numpy as np


device = torch.device('cuda')


# load model
model = models.resnet18(pretrained=True)
model = model.eval()  # put network in eval mode
model = model.to(device)

# image loader
r_mean, g_mean, b_mean = (0.485,  0.456, 0.406)
r_std, g_std, b_std = (0.229, 0.224, 0.225)
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(r_mean, g_mean, b_mean),
                         std=(r_std, g_std, b_std))
])

input_img = Image.open('../samples/dog.jpg')
input_img_tensor = img_transform(input_img).unsqueeze(0)

# showtensor(input_img_tensor.numpy())

input_img_tensor = input_img_tensor.to(device)

with torch.no_grad():
    predict = model.forward(input_img_tensor)

# make sure to get back prediction in CPU
predict = predict.to('cpu')

top10 = [reversed(predict[0].sort()[i][-10:].tolist()) for i in [0, 1]]
top10 = zip(*top10)

with open("imagenet1000_clsid_to_human.txt") as f:
    idx2label = eval(f.read())

for score, idx in top10:
    print(f'{score} : {idx2label[idx]} ({idx})')
