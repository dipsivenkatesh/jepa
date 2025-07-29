import torch.nn as nn
import torch

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    def loss(self, prediction, target):
        return nn.MSELoss()(prediction, target)
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()  # Set the model to evaluation mode after loading