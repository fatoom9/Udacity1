# **Image Classifier with Deep Learning**

The **Image Classifier** project is a deep learning-based application that classifies images into different categories using a pre-trained model. The model is trained on the **Oxford Flowers 102** dataset and utilizes **TensorFlow** and **Keras** for building and training the neural network. The application is designed to predict the class of input images, offering additional features like top-K predictions and category name mapping. The model uses **MobileNet** as the base model, fine-tuned for classifying flower species. It can predict multiple classes for each image, providing the top **K** predictions with probabilities, and the model is saved after training for future predictions.

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
To run the project, first clone the repository by using the following command:

```bash
git clone https://github.com/your-username/image-classifier.git
python train_model.py
python predict.py --image_path path_to_your_image --top_k 5
python predict.py --image_path path_to_your_image --category_names category_names.json
image-classifier/
│
├── train_model.py          # Script for training the model
├── predict.py              # Script for predicting class of an image
├── model.h5                # Saved trained model (generated after training)
├── category_names.json     # JSON file containing the mapping of class numbers to human-readable names
└── README.md               # This file
### **Contribution**

Feel free to fork this project, create issues, and submit pull requests. If you have suggestions or bug reports, please open an issue on GitHub.

### **License**
This project is licensed under the MIT License - see the LICENSE file for details.

### **Acknowledgments**
Thanks to the Oxford Flowers 102 dataset for providing the data used in training the model. Special thanks to Abdelrhman Wahdan for providing guidance and 
support throughout the project.
