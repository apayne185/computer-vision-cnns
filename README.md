# CNN Models for Image Classification

This repository contains two Jupyter notebooks that demonstrate building and evaluating convolutional neural network (CNN) models for image classification using TensorFlow and Keras.

##  Contents

### 1. `mnist-cnn.ipynb`

This notebook trains a CNN from scratch on the classic MNIST handwritten digit dataset. It includes:

- Loading and preprocessing the MNIST dataset
- Building a custom CNN architecture using Keras
- Compiling the model with optimizer, loss, and evaluation metrics
- Training the model and visualizing accuracy/loss
- Evaluating the model on test data

#### Output:
- Accuracy and loss plots
- Final test accuracy score

### 2. `test-cnns.ipynb`

This notebook explores the use of pretrained CNN architectures (such as Xception, ResNet34, and ResNet50) for transfer learning on custom datasets using Keras' applications module.

It includes:

- Loading pretrained models with ImageNet weights (`Xception`, etc.)
- Modifying the classifier head (Global Average Pooling + Dense layer)
- Fine-tuning the model by unfreezing layers
- Compiling with SGD + Nesterov momentum
- Fitting on a custom training and validation set

#### Output:
- Trained transfer learning model with accuracy/validation metrics

---

##  Requirements

To run the notebooks, install the following packages:

```bash
pip install tensorflow tensorflow-datasets scikit-learn matplotlib numpy pandas



##  Datasets

- **`mnist-cnn.ipynb`**  
  Uses the built-in **MNIST** dataset from `keras.datasets`.

- **`test-cnns.ipynb`**  
  Designed to be used with a **custom image dataset**.  
  You must define:
  - `train_set`
  - `valid_set`
  - `dataset_size`

---

##  Models Used

- **Custom CNN**  
  Built from scratch using basic layers:
  - `Conv2D`
  - `MaxPooling2D`
  - `Dense`

- **Pretrained CNNs (Transfer Learning)**  
  Leveraging pretrained weights from ImageNet:
  - `Xception`
  - `ResNet34` / `ResNet50` 

---

##  How to Run

1. Open the notebooks in **Jupyter Notebook** or **VS Code**.
2. Ensure all required packages are installed (`tensorflow`, `matplotlib`, `pandas`, etc.).
3. Run each cell step-by-step, or select **Run All** from the menu.

---