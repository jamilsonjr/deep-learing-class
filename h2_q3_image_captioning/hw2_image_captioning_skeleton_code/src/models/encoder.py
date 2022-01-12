import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        
        model_pretrained = models.resnet18(pretrained=True) #101
        modules = list(model_pretrained.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.encoder_dim = 512
        
        # Disable calculation of all gradients
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, images):

        out = self.model(images)  # (batch_size, 512, 16, 16)
    
        out = out.permute(0, 2, 3, 1) #([batch_size, 16, 16, 512])
        return out

