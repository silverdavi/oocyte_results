
# External
import torch.nn as nn
from torchvision import models
from torchvision.models.vision_transformer import ViT_B_16_Weights

class ViTBinaryClassifier(nn.Module):
    def __init__(self, hidden_dim = 512, dropout = 0.5):
        super(ViTBinaryClassifier, self).__init__()
        # Load pre-trained Vision Transformer
        self.vit = models.vit_b_16(weights = ViT_B_16_Weights.DEFAULT)
        # Optionally freeze ViT parameters to prevent them from being updated during training
        self.vit.heads = nn.Identity()
        for param in self.vit.parameters():
            param.requires_grad = False
        # Get the embedding dimension from ViT
        vit_embed_dim = self.vit.hidden_dim  # For vit_b_16, this is typically 768
        # Define the MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(vit_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Output a single logit for binary classification
            nn.Sigmoid()  # Output a single logit for binary classification
        )
    def forward(self, x):
        # Extract features using ViT
        embeddings = self.vit(x)  # Shape: [batch_size, embed_dim]
        # Pass embeddings through the MLP classifier
        logits = self.classifier(embeddings)  # Shape: [batch_size, 1]
        return logits  # Raw logits (use BCEWithLogitsLoss)
