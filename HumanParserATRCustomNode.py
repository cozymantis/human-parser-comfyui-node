import torch
import numpy as np
from PIL import Image

from .utils import generate

class HumanParserATRCustomNode:
  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "image" : ("IMAGE", {}),
        "background": ("BOOLEAN", {"default": False}),
        "hat": ("BOOLEAN", {"default": False}),
        "hair": ("BOOLEAN", {"default": False}),
        "sunglasses": ("BOOLEAN", {"default": False}),
        "upper_clothes": ("BOOLEAN", {"default": False}),
        "skirt": ("BOOLEAN", {"default": False}),
        "pants": ("BOOLEAN", {"default": False}),
        "dress": ("BOOLEAN", {"default": False}),
        "belt": ("BOOLEAN", {"default": False}),
        "left_shoe": ("BOOLEAN", {"default": False}),
        "right_shoe": ("BOOLEAN", {"default": False}),
        "face": ("BOOLEAN", {"default": False}),
        "left_leg": ("BOOLEAN", {"default": False}),
        "right_leg": ("BOOLEAN", {"default": False}),
        "left_arm": ("BOOLEAN", {"default": False}),
        "right_arm": ("BOOLEAN", {"default": False}),
        "bag": ("BOOLEAN", {"default": False}),
        "scarf": ("BOOLEAN", {"default": False}),
      },
    }

  RETURN_TYPES = ("MASK", "IMAGE")
  RETURN_NAMES = ("mask", "map")
  FUNCTION = "run"
  CATEGORY = "CozyMantis"

  def run(self, image, background, hat, hair, sunglasses, upper_clothes, skirt, pants, dress, belt, left_shoe, right_shoe, face, left_leg, right_leg, left_arm, right_arm, bag, scarf):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    output_img = generate(image[0], 'atr', device)

    mask_components = []

    if background:
      mask_components.append(0)
    if hat:
      mask_components.append(1)
    if hair:
      mask_components.append(2)
    if sunglasses:
      mask_components.append(3)
    if upper_clothes:
      mask_components.append(4)
    if skirt:
      mask_components.append(5)
    if pants:
      mask_components.append(6)
    if dress:
      mask_components.append(7)
    if belt:
      mask_components.append(8)
    if left_shoe:
      mask_components.append(9)
    if right_shoe:
      mask_components.append(10)
    if face:
      mask_components.append(11)
    if left_leg:
      mask_components.append(12)
    if right_leg:
      mask_components.append(13)
    if left_arm:
      mask_components.append(14)
    if right_arm:
      mask_components.append(15)
    if bag:
      mask_components.append(16)
    if scarf:
      mask_components.append(17)

    mask = np.isin(output_img, mask_components).astype(np.uint8)
    mask_image = Image.fromarray(mask * 255)
    mask_image = mask_image.convert("RGB")
    mask_image = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)

    output_img = output_img.convert('RGB')
    output_img = torch.from_numpy(np.array(output_img).astype(np.float32) / 255.0).unsqueeze(0)
    return (mask_image[:, :, :, 0], output_img,)
