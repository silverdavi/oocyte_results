
# Standard
import os
from typing import Iterable, List
# External
print('import PIL')
from PIL import Image, ImageFile
print('import torch')
import torch
from torch.nn import Module, ModuleList
print('import torchvision')
from torchvision.models import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights
from torchvision.transforms import Resize, ToTensor, Normalize, Compose

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.hub.set_dir(os.path.dirname(__file__))

FOLDS = 8
WEIGHTS_DIR = 'checkpoints'

print('init transform')
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(
        mean = (0.485, 0.456, 0.406),  # ImageNet mean
        std = (0.229, 0.224, 0.225),   # ImageNet std
    )
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

class Inferer(Module):
    def __init__(self, embedder: Module, classifiers: Iterable[Module]):
        super(Inferer, self).__init__()
        self.embedder = embedder
        self.classifiers = ModuleList(classifiers)
    def forward(self, x):
        embeddings = self.embedder(x)
        return torch.cat([ cls(embeddings) for cls in self.classifiers ])
    def predict(self, filepath: str) -> List[float]:
        image = Image.open(filepath).convert('RGB')
        input = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = self(input).squeeze().tolist()
            return preds

print('load vision transformer')
embedder = vit_b_16(weights = ViT_B_16_Weights.DEFAULT)
embedder.heads = torch.nn.Identity()

print('load classifiers')
kwargs = dict(map_location = device, weights_only = False)
classifiers = [ torch.load(f'{WEIGHTS_DIR}/cls.{fold}.pth', **kwargs) for fold in range(FOLDS) ]

print('init model')
model = Inferer(embedder, classifiers)
model.to(device)
model.eval()
