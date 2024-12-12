# **Image Classifier with Deep Learning**

## **Project Overview**

The **Image Classifier** project is a deep learning-based application that classifies images into different categories using a pre-trained model. The model is trained on the **Oxford Flowers 102** dataset and utilizes **TensorFlow** and **Keras** for building and training the neural network. The application is designed to predict the class of input images, offering additional features like top-K predictions and category name mapping.

## **Features**

- **Pre-trained Model**: Utilizes **MobileNet** as the base model, fine-tuned for classifying flower species.
- **Top-K Predictions**: The model can predict multiple classes for each input image, providing the top **K** predictions with probabilities.
- **Image Prediction**: Predicts the class of input images using a trained neural network.
- **Category Mapping**: Option to map class predictions to human-readable category names.
- **Model Saving and Loading**: The model is saved after training and can be loaded for future predictions.

## **Technologies Used**

- **Python** for building the deep learning model.
- **TensorFlow** and **Keras** for neural network implementation and training.
- **OpenCV** for image processing (optional, depending on the implementation).
- **NumPy** and **Matplotlib** for data manipulation and visualization.

## **Dataset**

The model is trained on the **Oxford Flowers 102** dataset, which contains **102** different flower categories with over **8,000** images.

- **Dataset link**: [Oxford Flowers 102 Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

## **Setup and Installation**

### **Prerequisites**

- **Python 3.x** installed on your machine.
- Install the required Python libraries by running the following command:

   ```bash
   pip install tensorflow numpy matplotlib opencv-python
