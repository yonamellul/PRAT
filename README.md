# Cervical Cancer Classification and Visualization with AI

This project focuses on the classification of cervical cancer using deep learning models and visualizes model decisions using Grad-CAM. It leverages modified ResNet-152 architecture for the task of classifying colposcopic images into low-risk and high-risk categories.

## Project Structure

The codebase is organized as follows:

- `loaders/`: Contains DataLoader objects saved in pickle format for easy loading during model training and evaluation.
- `models/`: Contains saved models in .pth format.
- `data_prep.py`: Scripts for data preprocessing and augmentation.
- `eval.py`: Contains functions for model evaluation.
- `gradCam.py`: Implementation of Grad-CAM for model visualization.
- `model.py`: Contains the modified ResNet-152 model definition.
- `training.py`: Scripts for model training.
- `utils.py`: Utility functions used across the project.
- `PRAT.ipynb`: A Jupyter notebook that provides an overview of the project and demonstrates the usage of Python functions.
- `README.md`: This file, providing an overview and instructions for the project.

## Getting Started

To get started with this project, follow these steps:

### Prerequisites

- Python 3.6 or later, along with pip for installing dependencies.

### Setup

1. **Clone the Repository:**

 ```sh
   git clone https://github.com/yonamellul/PRAT.git
   cd PRAT
 ```

2. **Unzip the PRAT_MELLUL/models Repository:**

3. **Install the required dependencies:**
```sh
  pip install -r requirements.txt
```

## Usage

To train and evaluate the model, run the `PRAT.ipynb` notebook. The notebook guides you through the process of loading data, training the model, and evaluating its performance. Additionally, it demonstrates how to use Grad-CAM for visualizing what the model has learned.
