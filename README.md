# Face Photo Generator Content Detector: Real vs AI-Generated Faces
## 📚 Description
This project aims to develop a content detection system that can differentiate between real human faces and AI-generated faces. With the rapid advancement of AI technologies in generating realistic images, it is crucial to create tools that can identify the authenticity of facial images. This project leverages computer vision techniques and deep learning models to provide an effective solution for detecting AI-generated faces.

## 📂 Dataset
The project leverages a diverse dataset that includes both real and AI-generated face images sourced from multiple Kaggle datasets. By carefully mixing and matching four distinct sources, the final dataset comprises 5,000 real images and 5,000 AI-generated images, organized into training, testing, and validation sets. This enhanced diversity has led to improved model performance, demonstrating the positive impact of having a varied dataset on classification accuracy.

### Sources:
  - [Real vs AI Generated Faces Dataset](https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset) 
  - [Detect AI-Generated Faces: High-Quality Dataset](https://www.kaggle.com/datasets/shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset) 
  - [AI Diverse Portraits](https://www.kaggle.com/datasets/youssefismail20/ai-male-and-female?select=AI+%28Males+%26+Females%29) 
  - [Stable Diffusion Face Dataset](https://www.kaggle.com/datasets/mohannadaymansalah/stable-diffusion-dataaaaaaaaa/data)

### Randomly Selected Images and Corresponding Model Classifications
<img width="560" height="312" alt="Screenshot 2025-11-21 at 5 12 32 AM" src="https://github.com/user-attachments/assets/4b17fc19-f0c7-48b7-b47d-8e5abe8ee43b" />


# First Model: Pytorch Convolutional Neural Network (CNN)
## 🏗️ Model Architecture Summary
- **Layer Types**: Various layers such as convolutional (Conv2d) and pooling (MaxPool2d) layers, which extract features from the input images.
- **Output Shapes**: The shape of data after each layer, demonstrating how the input size is transformed through the network.
- **Parameter Counts**: The number of parameters for each layer, highlighting the model's complexity and capacity for learning.
This architecture is specifically designed to efficiently process and classify facial images, enabling effective differentiation between real and AI-generated faces. The combination of multiple layers allows the model to capture intricate patterns and representations in the data.

<img width="573" height="348" alt="Screenshot 2025-11-22 at 11 42 40 AM" src="https://github.com/user-attachments/assets/f0d1e200-97fd-471f-9f7c-0b65f4686c76" />


## ⚙️ Technologies Used
- **Python**: The primary programming language for implementing the project.
- **Google Colab**: Used for accessing GPU resources to significantly accelerate model training.
- **PyTorch**: The deep learning framework utilized for building and training the model.
- **Libraries**: Torch, NumPy, Matplotlib, PIL (Pillow)


 
## 📈 Results
