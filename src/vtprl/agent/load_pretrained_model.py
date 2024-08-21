import torch
import torchvision.transforms as T
import torchvision
import numpy as np
from PIL import Image

from load_r3m import load_r3m

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def r3m_wrapper(rgb_image: np.ndarray, r3m_model) -> torch.Tensor:  # RGB_image should be np.array

    
    rgb_image = check_input_is_none(rgb_image)
    rgb_image = apply_transforms(rgb_image)
    rgb_image  = rgb_image.to(device)
   
    with torch.no_grad():
        state_embedding = r3m_model(rgb_image * 255.0)  # R3M expects image input to be [0-255]
        
    # For debugging
    # print(f"Embedding shape: {state_embedding.shape}")
    # print(f"Embedding type: {type(state_embedding)}")
    # print(f"Embedding device: {state_embedding.device}")

    return state_embedding
    
    
def check_input_is_none(rgb_image) -> np.ndarray:
    """Unity outputs images at the format H, W, C (e.g., 600x600x3)"""
    
    if rgb_image is None:
        rgb_image = np.zeros((600, 600, 3))
        return rgb_image
        
    elif isinstance(rgb_image, np.ndarray):
        return rgb_image
 
    elif isinstance(rgb_image, torch.tensor):
        raise ValueError("Rgb_image  expected np.ndarray not torch.tensor torch.tensor")

    else:
    	raise ValueError("Unknown rgb_image dtype, expected np.ndarray")
    
            
def get_r3m_model(r3m_type="resnet18"):
    model = load_r3m(r3m_type)  # resnet18, resnet34, resnet50
    model.eval()
    model = model.to(device)
    return model


def apply_transforms(rgb_image: np.ndarray) -> torch.tensor:
    """ Expects rgb_image as np.ndarray with (H,W,C) in range (0,255)"""
    transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()]) # ToTensor() divides by 255
    rgb_image = transforms(Image.fromarray(rgb_image.astype(np.uint8))).reshape(-1, 3, 224, 224)
    return rgb_image

