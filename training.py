from utils import *
from eval import *

CUDA = False
def main(modelname,  batch_size, unfreeze, num_epochs, lr, followup=None, augment = False, loss = 'CE'):
    """
    Main training loop for the model. Configures the model, data loaders, loss function, and optimizer,
    and performs training and validation.

    Parameters:
    - modelname (str): Name for saving the trained model.
    - batch_size (int): Batch size for training and validation.
    - unfreeze (int): Number of layers to unfreeze for fine-tuning.
    - num_epochs (int): Number of epochs to train the model.
    - lr (float): Learning rate for the optimizer.
    - followup (str, optional): Path to a checkpoint for loading a pre-trained model.
    - augment (bool): Whether to use data augmentation.
    - loss (str): Loss function to use ('CE' for CrossEntropyLoss, 'FL' for FocalLoss).
    """
    # Check for CUDA availability
    device = torch.device("cuda" if CUDA else "cpu")

    # Transfer Learning with ResNet-152
    model = ModifiedResNet152(num_classes=2, unfreeze_layers=unfreeze).to(device)
    if followup :
        checkpoint = torch.load('models/'+followup+'pth')
        model.load_state_dict(checkpoint)

    if augment:
        train_dataloader, validation_dataloader, test_dataloader = load_loaders('avec_aug')
    else:
        train_dataloader, validation_dataloader, test_dataloader = load_loaders('sans_aug')

    if CUDA: # If we have GPU, use CUDA
        cudnn.benchmark = True
        model = model.cuda()

    # Define loss function and optimizer
    if loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    if loss == 'FL':
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')

    optimizer = torch.optim.Adam(model.fc.parameters(), lr)

    # Model Training
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels, _ in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, _ in validation_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                total += labels.size(0)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item()}, Validation Accuracy: {100 * val_correct / total}%')

    save_model(model, modelname)

    return model



class FocalLoss(nn.Module):
    """
    Implementation of the Focal Loss, a variant of the binary cross-entropy loss that puts more focus
    on hard-to-classify examples. It is particularly useful for addressing class imbalance.

    Attributes:
    - alpha (float): The alpha parameter controls the weighting of the classes.
    - gamma (float): The gamma parameter controls the focusing parameter.
    - reduction (str): Specifies the reduction to apply to the output ('mean', 'sum', or 'none').
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, F.one_hot(targets, num_classes=2).float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_factor = F.one_hot(targets, num_classes=2).float() * self.alpha + (1 - F.one_hot(targets, num_classes=2).float()) * (1 - self.alpha)
        F_loss = alpha_factor * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

