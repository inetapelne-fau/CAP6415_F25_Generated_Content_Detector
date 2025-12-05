Reproducibility Instructions
To ensure that this project can be easily reproduced, follow the instructions below. All models should run smoothly with the provided code, utilizing the necessary libraries and imports listed at the beginning of the script.Google Colab with GPU
I developed this project using Google Colab, which provides a convenient cloud-based environment for running Python code. Here are a few highlights:
Free Access to GPUs: Colab gives me free access to powerful GPU resources, which really speeds up the training process for deep learning models. This is especially helpful for computationally intensive tasks like image classification.


Easy Setup: I love how quickly I can set up my environment without any local installation. Colab seamlessly integrates with popular libraries like PyTorch.


Interactive Notebooks: The Jupyter notebook interface allows me to write, execute, and visualize code and results interactively. This is perfect for experimentation and refining my work.


Collaboration: Sharing my Colab notebooks is super easy, making collaboration with others straightforward for team projects or peer reviews.

To enable GPU acceleration in Google Colab, just go to Runtime > Change runtime type and select GPU as the hardware accelerator.Models Overview
The project includes the following five models:
Custom CNN
ResNet-18
EfficientNet-B0
Vision Transformer (ViT) B-16
Model A (Note: requires user input for saving)

Running the Models

Environment Setup: 
I recommend using Google Colab for running the code. It provides free access to GPUs, which will significantly speed up the training process.




Library Imports: 
All necessary libraries and imports are included at the beginning of the script. Ensure that you have the following libraries installed if you are running it locally:
PyTorch
NumPy
Matplotlib
Other relevant libraries as needed.





Special Instructions for Model A

When running Model A, you will be prompted to provide input for saving the model. Make sure to specify the desired file path and name when prompted, as this model requires user interaction for saving.

Here’s an example of how the input might look when saving Model A:Code# Example input promptmodel_save_path = input("Please enter the path where you want to save the model: ")torch.save(model.state_dict(), model_save_path)By following these instructions, you should be able to reproduce the results of this project successfully. If you encounter any issues, feel free to reach out for assistance!
