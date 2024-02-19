from utils import *


# Transformation pipeline to apply to the images.
transform = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def augment_images(train_pids, dataset_path="../data_aug", class_folders=['0', '1'], augment_factor={'0': 1, '1': 3}):
    """
    Apply data augmentation techniques to images of a dataset. This includes flipping images
    horizontally, vertically, or both, and applying random rotations. The augmented images are
    saved back to the dataset.

    Args:
    train_pids (list): A list of patient IDs to apply augmentation on.
    dataset_path (str): The directory path where the dataset is stored.
    class_folders (list): A list containing the names of subfolders for each class in the dataset.
    augment_factor (dict): A dictionary indicating the number of times to augment each class.
    """
    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        images = os.listdir(class_path)

        for image_name in images:

            patient_id = int(image_name.split('_')[0])
            print("pid = ", patient_id)
            if patient_id not in train_pids:
                continue  # Skip images not in the training set

            image_path = os.path.join(class_path, image_name)
            image = Image.open(image_path)

            # Define flip types
            flip_actions = [(Image.FLIP_LEFT_RIGHT, "h"), (Image.FLIP_TOP_BOTTOM, "v"), ("hv", "hv")]
            rotations = [36, 72, 108, 144, 216, 252, 288, 324]

            # Apply all flip actions if augment factor is greater than 3
            if augment_factor[class_folder] > 3:
                for flip_type, suffix in flip_actions:
                    flipped_image = apply_flip(image, flip_type)
                    save_flipped_image(flipped_image, image_name, suffix, class_path)

                # Additional random rotations for remaining augmentations
                random.shuffle(rotations)
                selected_rotations = rotations[:augment_factor[class_folder]-3]
                for angle in selected_rotations:
                    #angle = random.choice([90, 180, 270])
                    rotated_image = image.rotate(angle)
                    suffix = f"rot{angle}"
                    save_flipped_image(rotated_image, image_name, suffix, class_path)
            else:
                # Apply flips for augment factors of 3 or less
                random.shuffle(flip_actions)
                selected_actions = flip_actions[:augment_factor[class_folder]]
                for flip_type, suffix in selected_actions:
                    flipped_image = apply_flip(image, flip_type)
                    save_flipped_image(flipped_image, image_name, suffix, class_path)

def apply_flip(image, flip_type):
    """
    Apply flipping to an image either horizontally, vertically, or both.

    Args:
    image (PIL.Image): The image to flip.
    flip_type (str or int): The type of flip to apply. Can be a string 'hv' for both horizontal
                            and vertical or an integer corresponding to PIL flip constants.

    Returns:
    PIL.Image: The flipped image.
    """
    if flip_type == "hv":  # Flip both ways
        return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    else:
        return image.transpose(flip_type)

def save_flipped_image(flipped_image, image_name, suffix, class_path):
    """
    Save an augmented image to the file system with a new filename.

    Args:
    flipped_image (PIL.Image): The augmented image to save.
    image_name (str): The original filename of the image.
    suffix (str): A string to append to the filename to indicate the type of augmentation.
    class_path (str): The directory path to save the augmented image.
    """
    new_image_name = f"{os.path.splitext(image_name)[0]}_{suffix}{os.path.splitext(image_name)[1]}"
    new_image_path = os.path.join(class_path, new_image_name)
    flipped_image.save(new_image_path)


def show_images_from_dataloader(dataloader, num_images=4):
    """
    Display a batch of images from a DataLoader.

    Args:
    dataloader (DataLoader): The DataLoader to get the images from.
    num_images (int): The number of images to display.
    """
    images, labels, pid = next(iter(dataloader))
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
    for i in range(num_images):
        ax = axes[i]
        img = images[i].permute(1, 2, 0)  # Convert from CxHxW to HxWxC format
        ax.imshow(img)
        ax.axis('off')
    plt.show()


def unique_pids(dataloader):
    """
    Extract unique patient IDs and their corresponding labels from a DataLoader.

    Args:
    dataloader (DataLoader): The DataLoader containing the dataset.

    Returns:
    set: A set containing unique patient IDs.
    dict: A dictionary mapping each unique patient ID to their corresponding label.
    """
    unique_pids = set()
    unique_labels= {}
    for _, labels, pids in dataloader:
        # Convert each pid tensor to integer and add to the set
        for label, pid in zip(labels,pids):
            unique_pids.update({pid.item()})
            if pid.item() not in unique_labels:
                unique_labels[pid.item()] = label.item()
    return unique_pids, unique_labels

class PatientDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset that loads images along with their labels and patient IDs.
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve an image, its label, and the patient ID at the given index, applying a transform if defined.

        Args:
        idx (int): The index of the item to retrieve.

        Returns:
        tuple: (transformed image, label, patient ID)
        """

        image, label = self.dataset[idx]
        patient_id = int(os.path.basename(self.dataset.imgs[idx][0]).split('_')[0])

        if self.transform:
            image = self.transform(image)

        return image, label, patient_id

def prepare_data(batch_size, CUDA, root="../"):
    """
    Prepares DataLoader objects for training, validation, and testing phases of a machine learning model. This function:
    - Applies standard preprocessing transformations to the dataset images.
    - Splits the dataset into training, validation, and testing sets based on unique patient IDs.
    - Performs data augmentation on the training set to enhance model robustness against overfitting.
    - Saves augmented images to disk for later use.
    - Creates DataLoader objects for each dataset subset, facilitating efficient batch processing during model training.

    Parameters:
    - batch_size (int): The size of each data batch to be loaded during model training.
    - CUDA (bool): Flag indicating whether GPU acceleration is available and should be utilized.
    - root (str): The root directory path where the datasets (both original and augmented) are stored.
    
    Saves:
    DataLoaders for the training, validation, and testing sets both with and without data augmentation.
    """
    
    # Load the dataset without augmentation from a specified root directory.
    full_dataset = ImageFolder(root=root + 'data')
    # Wrap the dataset with the PatientDataset class, applying the defined transformations.
    patient_dataset = PatientDataset(full_dataset, transform)

    # Extract patient IDs and labels from the dataset for use in train/test splits.
    patient_ids = [int(os.path.basename(img[0]).split('_')[0]) for img in full_dataset.imgs]
    labels = [img[1] for img in full_dataset.imgs]
    unique_patient_ids = list(set(patient_ids))
    unique_labels = [labels[patient_ids.index(pid)] for pid in unique_patient_ids]

    # Split the unique patient IDs into training and testing sets, maintaining class balance.
    train_ids, test_ids = train_test_split(unique_patient_ids, stratify=unique_labels, test_size=0.15)

    # Perform data augmentation on the training set and save the augmented images to disk.
    class_folders = ['0', '1']
    augment_factor = {'0': 0, '1': 7}
    augment_images(train_ids, class_folders=class_folders , augment_factor=augment_factor)

    # Further split the training set into training and validation sets.
    train_ids, val_ids = train_test_split(train_ids, stratify=[unique_labels[unique_patient_ids.index(pid)] for pid in train_ids], test_size=0.1)

    # Create subsets for training, validation, and testing using the original patient dataset (without augmentation).
    train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_ids]
    test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_ids]
    val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_ids]

    train_dataset = Subset(patient_dataset, train_indices)
    val_dataset = Subset(patient_dataset, val_indices)
    test_dataset = Subset(patient_dataset, test_indices)
    
    # Prepare DataLoader objects for each dataset subset for efficient data loading during training.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=CUDA)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=CUDA)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=CUDA)
    
    # Save the DataLoader objects for easy loading in future sessions.
    save_loaders('sans_aug',train_dataloader, validation_dataloader, test_dataloader)

    # Repeat the process for the dataset with augmentation.
    full_dataset = ImageFolder(root=root + 'data_aug')
    patient_dataset = PatientDataset(full_dataset, transform)

    patient_ids = [int(os.path.basename(img[0]).split('_')[0]) for img in full_dataset.imgs]
    labels = [img[1] for img in full_dataset.imgs]
    #unique_patients_ids and unique_labels stay the same, no need to compute them again

    train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_ids]
    test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_ids]
    val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_ids]

    train_dataset = Subset(patient_dataset, train_indices)
    val_dataset = Subset(patient_dataset, val_indices)
    test_dataset = Subset(patient_dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=CUDA)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=CUDA)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=CUDA)

    save_loaders('avec_aug',train_dataloader, validation_dataloader, test_dataloader)



'''
def prepare_data(batch_size, CUDA, root="../"):
    transform = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    #prepare the loaders from the dataset with no augmentation whose name is 'data'
    full_dataset = ImageFolder(root=root + 'data')
    patient_dataset = PatientDataset(full_dataset, transform)

    patient_ids = [int(os.path.basename(img[0]).split('_')[0]) for img in full_dataset.imgs]
    labels = [img[1] for img in full_dataset.imgs]
    unique_patient_ids = list(set(patient_ids))
    unique_labels = [labels[patient_ids.index(pid)] for pid in unique_patient_ids]

    #train_ids, test_ids = train_test_split(unique_patient_ids, stratify=unique_labels, test_size=0.15)
    train_ids, test_ids, val_ids = list(train_dic.keys()), list(test_dic.keys()), list(val_dic.keys())

    train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_ids]
    test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_ids]
    val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_ids]

    train_dataset = Subset(patient_dataset, train_indices)
    val_dataset = Subset(patient_dataset, val_indices)
    test_dataset = Subset(patient_dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=CUDA)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=CUDA)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=CUDA)

    save_loaders('sans_aug',train_dataloader, validation_dataloader, test_dataloader)

    #now we prepare the loaders with augmentation
    full_dataset = ImageFolder(root=root + 'data_aug')
    patient_dataset = PatientDataset(full_dataset, transform)

    patient_ids = [int(os.path.basename(img[0]).split('_')[0]) for img in full_dataset.imgs]
    labels = [img[1] for img in full_dataset.imgs]
    #unique_patients_ids and unique_labels stay the same, no need to recompute them

    train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_ids]
    test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_ids]
    val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_ids]

    train_dataset = Subset(patient_dataset, train_indices)
    val_dataset = Subset(patient_dataset, val_indices)
    test_dataset = Subset(patient_dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=CUDA)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=CUDA)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=CUDA)

    save_loaders('avec_aug',train_dataloader, validation_dataloader, test_dataloader)

'''