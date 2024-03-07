import torch
import numpy as np
from PIL import Image

from .utils import generate
['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs']
class HumanParserPascalCustomNode:
  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image" : ("IMAGE", {}),
        "background": ("BOOLEAN", {"default": False}),
        "head": ("BOOLEAN", {"default": False}),
        "torso": ("BOOLEAN", {"default": False}),
        "upper_arms": ("BOOLEAN", {"default": False}),
        "lower_arms": ("BOOLEAN", {"default": False}),
        "upper_legs": ("BOOLEAN", {"default": False}),
        "lower_legs": ("BOOLEAN", {"default": False}),
      },
    }

  RETURN_TYPES = ("MASK", "IMAGE")
  RETURN_NAMES = ("mask", "map")
  FUNCTION = "run"
  CATEGORY = "CozyMantis"

  def run(self, image, background, head, torso, upper_arms, lower_arms, upper_legs, lower_legs):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    output_img = generate(image[0], 'pascal', device)

    mask_components = []

    if background:
      mask_components.append(0)
    if head:
      mask_components.append(1)
    if torso:
      mask_components.append(2)
    if upper_arms:
      mask_components.append(3)
    if lower_arms:
      mask_components.append(4)
    if upper_legs:
      mask_components.append(5)
    if lower_legs:
      mask_components.append(6)

    mask = np.isin(output_img, mask_components).astype(np.uint8)
    mask_image = Image.fromarray(mask * 255)
    mask_image = mask_image.convert("RGB")
    mask_image = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)

    output_img = output_img.convert('RGB')
    output_img = torch.from_numpy(np.array(output_img).astype(np.float32) / 255.0).unsqueeze(0)
    return (mask_image[:, :, :, 0], output_img,)
