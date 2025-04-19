import cv2
import torch
import os
import numpy as np
from collections import OrderedDict
import torchvision.transforms as transforms
from PIL import Image

from .schp import networks
from .schp.utils.transforms import transform_logits, get_affine_transform

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def _box2cs(box, aspect_ratio):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, aspect_ratio)

def _xywh2cs(x, y, w, h, aspect_ratio):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32)
    return center, scale

def check_model_path(model_path):
    # Checks to see if the model exists, if not try adding ComfyUI/ to the start to fix possible errors on Windows (maybe others too)
    if not os.path.exists(model_path):
        new_model_path = os.path.join("ComfyUI", model_path)
        if os.path.exists(new_model_path):
            return new_model_path
    return model_path

def generate(image, type, device):
  num_classes = dataset_settings[type]['num_classes']
  input_size = dataset_settings[type]['input_size']
  aspect_ratio = input_size[1] * 1.0 / input_size[0]
  if type == 'lip':
    model_path = 'models/schp/exp-schp-201908261155-lip.pth'
  elif type == 'atr':
    model_path = 'models/schp/exp-schp-201908301523-atr.pth'
  elif type == 'pascal':
    model_path = 'models/schp/exp-schp-201908270938-pascal-person-part.pth'

  # Check and adjust the model path if necessary
  model_path = check_model_path(model_path)

  model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
  state_dict = torch.load(model_path)['state_dict']
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
      name = k[7:]
      new_state_dict[name] = v
  model.load_state_dict(new_state_dict)
  model.to(device)
  model.eval()

  # Get person center and scale
  input = 255. * image.cpu().numpy()
  input = np.clip(input, 0, 255).astype(np.uint8)
  input = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
  h, w, _ = input.shape

  person_center, s = _box2cs([0, 0, w - 1, h - 1], aspect_ratio)
  trans = get_affine_transform(person_center, s, 0, input_size)
  input = cv2.warpAffine(
      input,
      trans,
      (int(input_size[1]), int(input_size[0])),
      flags=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_CONSTANT,
      borderValue=(0, 0, 0))

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
  ])
  input = transform(input)

  palette = get_palette(num_classes)
  with torch.no_grad():
    input = input[None, :, :, :]
    output = model(input.to(device))
    upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    upsample_output = upsample(output[0][-1][0].unsqueeze(0))
    upsample_output = upsample_output.squeeze()
    upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

    logits_result = transform_logits(upsample_output.data.cpu().numpy(), person_center, s, w, h, input_size=input_size)
    parsing_result = np.argmax(logits_result, axis=2)

    output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
    output_img.putpalette(palette)
  return output_img
