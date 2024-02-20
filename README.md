## Car Brand Prediction Project

Welcome to the Car Brand Prediction Project! This project utilizes image classification techniques to predict different car brands from input images. It employs the Keras framework with ResNet50 as the backbone for training and inference.

### Features

- Image Classification: The project classifies input images into various car brands.
- ResNet50 Backbone: ResNet50, a powerful convolutional neural network architecture, is used as the backbone for feature extraction and classification.
- Transfer Learning: Transfer learning is employed to fine-tune the pre-trained ResNet50 model on the specific task of car brand classification.
- Easy to Use: The project provides a straightforward interface for users to input images and receive predictions for the corresponding car brands.


### Installation

Clone the repository to your local machine:

```
git clone https://github.com/your-username/car-brand-prediction.git
```
Navigate to the project directory:

```
cd car-brand-prediction
```

Install the required dependencies:
```
pip install -r requirements.txt
```

### Dataset
The dataset used for training and testing the model consists of images of various car brands. It is not included in this repository due to size limitations. However, you can use any suitable dataset for training and testing the model.

### Usage
- Prepare your dataset or use the provided dataset.
- Train the model using the train.py script, specifying the dataset directory and other parameters as needed.
- After training, use the trained model for inference by running the predict.py script and providing the path to the input image.
- Alternatively, deploy the model in a web application or any other suitable environment for real-time predictions.


### Model Evaluation
Evaluate the performance of the trained model using metrics such as accuracy, precision, recall, and F1-score on a separate validation dataset. Adjust hyperparameters and model architecture as necessary to improve performance.

### Dependencies
- Keras
- TensorFlow
- ResNet50
