# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import omegaconf
import hydra
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from r3m import load_r3m

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


### Call R3M
r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval()
r3m.to(device)

### DEFINE PREPROCESSING
transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()]) # ToTensor() divides by 255

### ENCODE IMAGE

## Generate a random np image
#image = np.random.randint(0, 255, (500, 500, 3))

## Use the example images in folder
#image_JPG = Image.open('image_1.jpg')
image_JPG = Image.open('image_2.jpg')
image = np.array(image_JPG)

### Show the data of image
print(image.shape)
plt.imshow(image)
plt.axis('off')  # Turn off axis
plt.show()

### Transform the image to the shape of R3M input
preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
print(preprocessed_image.shape)


### Show the data of transformed image
tensor = preprocessed_image.permute(0, 2, 3, 1)
adjusted_tensor = tensor.squeeze(dim=0)
print(adjusted_tensor.shape)
plt.imshow(adjusted_tensor)
plt.axis('off')  # Turn off axis
plt.show()


### R3M
preprocessed_image.to(device)
with torch.no_grad():
  embedding = r3m(preprocessed_image * 255.0) ## R3M expects image input to be [0-255]
print(embedding.type) # [1, 2048]


### Visualize the Output of R3M
reshape_embedding = embedding.reshape(32, 64)
print(reshape_embedding)
plt.imshow(reshape_embedding)
plt.axis('off')  # Turn off axis
plt.show()
