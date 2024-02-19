from utils import *

class ModifiedResNet152(nn.Module):
    """
    A modified version of the ResNet-152 model adapted for transfer learning on a custom dataset.
    This class allows for the modification of the number of output classes and the ability to unfreeze
    the last few layers of the model for fine-tuning.

    Attributes:
    - num_classes (int): The number of output classes for the model.
    - unfreeze_layers (int): The number of layers from the end to unfreeze for training.
    """
    def __init__(self, num_classes=2, unfreeze_layers=3):
        super(ModifiedResNet152, self).__init__()
        # Load the pretrained ResNet-152 model
        original_model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)

        # Freeze every layer
        for param in original_model.parameters():
            param.requires_grad = False
        # Unfreeze the specified number of layers from the end.
        for layer in list(original_model.children())[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Replace the original fully connected layer with a new one for the specified number of classes.
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.flatten = nn.Flatten()
        num_ftrs = original_model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.flatten(x)
        x = self.fc(x)
        return x