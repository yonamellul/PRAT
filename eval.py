from utils import *
from gradCam import *

def evaluate_model(model_name, device, augment = False , voting='majority'):
    """
    Evaluates the performance of a given model on a test dataset, optionally using data augmentation.
    It aggregates predictions at the patient level, allowing for majority voting or other custom aggregation methods.
    
    Parameters:
    - model_name (str): The name of the model to be loaded and evaluated.
    - device (torch.device): The device (CPU or GPU) to perform the evaluation on.
    - augment (bool): Flag to determine whether the model used augmented data for training.
    - voting (str): The strategy for aggregating predictions across multiple images of the same patient.
    
    Returns:
    - A set of evaluation metrics including accuracy, AUC, sensitivity, specificity, PPV, and NPV.
    """

    model = load_model(model_name)
    model.eval()

    patient_predictions = {}
    patient_true_labels = {}
    patient_scores = {}  # Collect scores for ROC


    if augment :
        _, _, data_loader = load_loaders('avec_aug')
    else:
        _, _, data_loader = load_loaders('sans_aug')


    #print('nb of unique pids: ', len(unique_pids(data_loader)))
    with torch.no_grad():
        for images, labels, patient_ids in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            scores = torch.softmax(outputs, dim=1)[:, 1]  # Softmax scores for class 1

            for pid, pred, score, label in zip(patient_ids, preds, scores, labels):
                patient_predictions.setdefault(pid.item(), []).append(pred.cpu().item())
                patient_scores.setdefault(pid.item(), []).append(score.cpu().item())  # Store scores
                patient_true_labels[pid.item()] = label.cpu().item()  # Assuming same label for all images of a patient


    y_true, y_pred, y_scores = [], [], []
    for pid, predictions in patient_predictions.items():
        patient_class = vote(predictions, mode=voting)
        y_pred.append(patient_class)
        y_true.append(patient_true_labels[pid])
        y_scores.append(np.mean(patient_scores[pid]))  # Use mean score for ROC


    # Compute metrics
    accuracy = (np.array(y_pred) == np.array(y_true)).mean()
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    # Compute precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    print(f'Patient-level Metrics:\nAccuracy: {accuracy * 100:.2f}%\nAUC: {auc:.2f}\nAP: {average_precision:.2f}\nSensitivity: {sensitivity * 100:.2f}%\nSpecificity: {specificity * 100:.2f}%\nPPV: {ppv * 100:.2f}%\nNPV: {npv * 100:.2f}%')

    # Plot ROC curve
    plot_roc_curve(auc, fpr, tpr)

    # Plot precision-recall curve
    plot_pr_curve(average_precision, recall, precision)

    # Display the confusion matrix as a heatmap
    display_conf_mat(cm)

    return accuracy, auc,  sensitivity, specificity, ppv, npv

def plot_roc_curve(auc, fpr, tpr):
    """
    Plots the ROC curve given the AUC, false positive rate (FPR), and true positive rate (TPR).
    
    Parameters:
    - auc (float): The area under the ROC curve.
    - fpr (array): False positive rate values.
    - tpr (array): True positive rate values.
    """
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_pr_curve(average_precision, recall, precision):
    """
    Plots the Precision-Recall curve given the average precision, recall, and precision values.
    
    Parameters:
    - average_precision (float): The average precision score.
    - recall (array): The recall values.
    - precision (array): The precision values.
    """

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
    plt.show()

def display_conf_mat(cm):
    """
    Displays the confusion matrix as a heatmap.
    
    Parameters:
    - cm (array): The confusion matrix to be displayed.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low-Risk", "High-Risk"])  # Replace with your class labels
    disp.plot(cmap=plt.cm.Blues, values_format=".4g")  # Adjust the colormap and format as needed
    plt.title("Confusion Matrix")
    plt.show()

def evaluate_all_patients(modelname, dataloader, device,data='test', voting='high', gradCam=False, only_high_risks=True):
    """
    Evaluates model predictions for all patients, optionally using Grad-CAM for visualization.
    Can filter to evaluate only high-risk patients.
    
    Parameters:
    - modelname (str): The name of the model to evaluate.
    - dataloader (DataLoader): DataLoader containing the dataset for evaluation.
    - device (torch.device): The device to perform evaluation on.
    - data (str): Specifies the dataset split to evaluate ('test', 'val', or 'train').
    - voting (str): Voting strategy for aggregating predictions.
    - gradCam (bool): Whether to use Grad-CAM for visualization.
    - only_high_risks (bool): Whether to evaluate only high-risk patients.
    """

    model = load_model(modelname).to(device)
    model.eval()
    target_layer = get_last_conv_name(model)

    # Extract unique patient IDs
    if data == 'test':
        uniqueLabels = test_dic
    if data == 'val':
        uniqueLabels = val_dic
    if data == 'train':
        uniqueLabels = train_dic
    uniquePids=list(uniqueLabels.keys())

    for pid, label in uniqueLabels.items():
        if label == 1 or not only_high_risks:
            voted_label, ground_truth = predict_patient(modelname, device, pid, dataloader, voting=voting)
            print(f"Patient ID : {pid} | Voted classification: {labels_bin2_dic[voted_label]} | Ground Truth: {labels_bin2_dic[ground_truth]}")

            if gradCam:
            # Grad-CAM and plot
                plot_image_with_gradcam(dataloader, modelname, pid, device)


def vote(predictions, mode='high'):
    """
    Aggregates image-level predictions to patient-level predictions based on a specified voting strategy.
    
    Parameters:
    - predictions (list): List of predictions for images of a single patient.
    - mode (str): Voting strategy ('majority' or 'high' risk preference).
    
    Returns:
    - The aggregated patient-level prediction.
    """

    if mode == 'majority':
    # Special rule for 2 images with opposing votes
        if len(predictions) == 2 and len(set(predictions)) == 2:
            patient_class = 1  # Default to high-risk
        else:
            patient_class = max(set(predictions), key=predictions.count)
    else: #If any of the predictions is high-risk (1), classify the patient as high-risk
        patient_class = 1 if 1 in predictions else 0
    return patient_class

def predict_patient(model_name, device, pid, dataloader, voting='high'):
    """
    Predicts the class for a single patient based on images and applies a voting mechanism if specified.
    
    Parameters:
    - model_name (str): The name of the model for prediction.
    - device (torch.device): The device for computation.
    - pid (int): Patient ID for which the prediction is to be made.
    - dataloader (DataLoader): DataLoader containing the dataset.
    - voting (str): Voting strategy to aggregate multiple image predictions.
    
    Returns:
    - The voted class label and the ground truth label for the patient.
    """
    model = load_model(model_name).to(device)
    model.eval()

    patient_images = []
    patient_labels = []

    # Iterate through dataloader to find all images for the given patient ID
    for images, labels, pids in dataloader:
        for img, label, patient_id in zip(images, labels, pids):
            if patient_id.item() == pid:
                patient_images.append(img.to(device))
                patient_labels.append(label.item())

    # If no images found for the patient, return
    if not patient_images:
        print(f"No images found for patient ID {pid}")
        return

    # Predict and apply voting if necessary
    predictions = []
    i = 1
    for img in patient_images:
        output = model(img.unsqueeze(0))
        pred = torch.argmax(output, dim=1).item()
        #print(f'pred{i} = {pred}')
        predictions.append(pred)
        i+=1
    
    return vote(predictions, mode=voting), patient_labels[0]


