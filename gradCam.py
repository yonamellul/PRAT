from utils import *


def get_last_conv_name(net):
    """
    Gets the name of the last convolutional layer in the given PyTorch model.

    Parameters:
    - net (torch.nn.Module): The model from which to find the last convolutional layer.

    Returns:
    - layer_name (str): The name of the last convolutional layer in the model.
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            layer_name = name
    return layer_name

def grad_cam(model, input_image, target_layer, device):
    """
    Implements the Grad-CAM algorithm to generate a heatmap highlighting the regions of the input image
    that are most important for the model's decision.

    Parameters:
    - model (torch.nn.Module): The model being interpreted.
    - input_image (torch.Tensor): The input image for which to generate the Grad-CAM heatmap.
    - target_layer (str): The name of the target layer to use for Grad-CAM.
    - device (torch.device): The device on which the computation is performed.

    Returns:
    - cam (torch.Tensor): The generated Grad-CAM heatmap as a tensor.
    """
    model.eval()
    target_layer = dict([*model.named_modules()])[target_layer]

    gradients = []
    activations = []
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        return None

    def forward_hook(module, input, output):
        activations.append(output)
        return None

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward pass

    if input_image.dim() == 4 and input_image.shape[0] == 1:
      input_image = input_image.squeeze(0)

    input_image = input_image.unsqueeze(0)
    output = model(input_image)
    if isinstance(output, tuple):
        output = output[0]

    # Target for backprop
    one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
    one_hot_output[0][torch.argmax(output)] = 1
    one_hot_output = one_hot_output.to(device)  # Ensure it's on the same device as the model

    # Backward pass
    model.zero_grad()
    output.backward(gradient=one_hot_output, retain_graph=True)

    # Generate cam map
    gradient = gradients[0].detach()
    activation = activations[0].detach()
    b, k, u, v = gradient.size()

    alpha = gradient.view(b, k, -1).mean(2)
    weights = alpha.view(b, k, 1, 1)

    cam = torch.relu(torch.sum(weights * activation, dim=1, keepdim=True))
    cam = F.interpolate(cam, input_image.shape[2:], mode='bilinear', align_corners=False)
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam

def show_cam_on_image(img, mask):
    """
    Overlays the Grad-CAM heatmap on the original image to visualize the regions of interest.

    Parameters:
    - img (numpy.ndarray): The original image as a NumPy array.
    - mask (numpy.ndarray): The Grad-CAM heatmap to overlay on the image.
    """
    heatmap = np.uint8(255 * mask)  # Convert mask to a heatmap
    heatmap = np.stack((heatmap,) * 3, axis=-1)  # Make it 3 channel to overlay
    cam_img = heatmap * 0.3 + img.numpy() * 0.5  # Overlay heatmap on image
    cam_img = cam_img / np.max(cam_img)  # Normalize for display

    plt.imshow(cam_img)
    plt.axis('off')
    plt.show()


def plot_image_with_gradcam(dataloader, model_name, patient_id,device):
    """
    Plots images for a specific patient with Grad-CAM heatmaps overlaid to highlight important regions
    contributing to the model's predictions.

    Parameters:
    - dataloader (DataLoader): The DataLoader containing the dataset.
    - model_name (str): The name of the model to evaluate.
    - patient_id (int): The patient ID for which to plot images with Grad-CAM heatmaps.
    - device (torch.device): The device on which the model is evaluated.
    """
    model = load_model(model_name).to(device)
    target_layer = get_last_conv_name(model)

    # Move model to evaluation mode
    model.eval()
    images_to_plot = []

    for images, labels, pids in dataloader:
        for img, label, pid in zip(images, labels, pids):
            if pid.item() == patient_id:
                images_to_plot.append((img, label))
                if len(images_to_plot) >= 4:
                    break
        if len(images_to_plot) >= 4:
            break

    if not images_to_plot:
        raise ValueError(f"No images found for patient ID {patient_id}.")

    # Set up subplots
    fig, axes = plt.subplots(1, len(images_to_plot), figsize=(15, 5))
    if len(images_to_plot) == 1:
      axes = [axes]

    for idx, (img, label) in enumerate(images_to_plot):
        image = img.unsqueeze(0).to(device)
        predicted_output = model(image)
        predicted_label = torch.argmax(predicted_output, dim=1).item()

        cam = grad_cam(model, image, target_layer, device)

        img_for_plot = image.cpu().squeeze().permute(1, 2, 0).numpy()
        img_for_plot = (img_for_plot - img_for_plot.min()) / (img_for_plot.max() - img_for_plot.min())

        cam_for_plot = cam.cpu().squeeze().numpy()
        cam_for_plot = (cam_for_plot - cam_for_plot.min()) / (cam_for_plot.max() - cam_for_plot.min())

        axes[idx].imshow(img_for_plot)
        axes[idx].imshow(cam_for_plot, cmap='jet', alpha=0.5)
        axes[idx].set_title(f'[{patient_id}] , classified as: {labels_bin2_dic[predicted_label]}')
        axes[idx].axis('off')

    plt.show()



