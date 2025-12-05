# AI Face Detection: Navigating the Challenges of Real and Synthetic Images
# 📚 Project Description
This project focuses on developing a robust content detection system that accurately distinguishes between real human faces and AI-generated images. As AI technologies continue to advance rapidly, creating highly realistic visuals, the potential for misuse increases, raising significant concerns about misinformation, privacy, and trust in digital media.

**Goal**: The primary goal of this project is to create a reliable tool that can effectively identify the authenticity of facial images. By doing so, we aim to protect individuals and society from the negative consequences of manipulated visuals, such as identity theft, deepfake videos, and the spread of false information.

**Problem**: The challenge lies in the increasing sophistication of AI-generated images, which can easily deceive even the most discerning viewers. As these technologies evolve, the distinction between genuine and fabricated content becomes increasingly blurred, making it vital to develop systems that can detect and mitigate these risks.

**Real-Life Usefulness**: This detection system will have practical applications across various fields, including:
- Security: Enhancing biometric verification systems by ensuring that only real human faces are authenticated.
- Media Integrity: Supporting journalists and content creators in verifying the authenticity of images, thereby preserving the credibility of information shared with the public.
- Social Media: Assisting platforms in identifying and removing deceptive content, fostering a safer online environment for users.

By addressing these challenges, this project aims to contribute to a more transparent and trustworthy digital landscape, empowering individuals and organizations to navigate the complexities of AI-generated content responsibly.

# 📂 Dataset Description
The project utilizes a carefully curated and diverse dataset that features both real and AI-generated face images, sourced from multiple Kaggle datasets. By strategically combining four distinct sources, the final dataset consists of 5,000 authentic images and 5,000 AI-generated images, systematically organized into training, testing, and validation sets. This thoughtful approach to dataset creation enhances diversity, which has been instrumental in improving model performance. The varied representations of facial features, expressions, and backgrounds within the dataset have significantly contributed to increased classification accuracy, highlighting the importance of a rich and varied dataset in training effective machine learning models.

#### Sources:
  - [Real vs AI Generated Faces Dataset](https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset) 
  - [Detect AI-Generated Faces: High-Quality Dataset](https://www.kaggle.com/datasets/shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset) 
  - [AI Diverse Portraits](https://www.kaggle.com/datasets/youssefismail20/ai-male-and-female?select=AI+%28Males+%26+Females%29) 
  - [Stable Diffusion Face Dataset](https://www.kaggle.com/datasets/mohannadaymansalah/stable-diffusion-dataaaaaaaaa/data)

#### Randomly Selected Images and Corresponding Model Classifications
<img width="560" height="312" alt="Screenshot 2025-11-21 at 5 12 32 AM" src="https://github.com/user-attachments/assets/4b17fc19-f0c7-48b7-b47d-8e5abe8ee43b" />

# 🛠️ Approaches
This project employs three distinct approaches to develop the content detection system for differentiating between real human faces and AI-generated images:

## 1. **Model A: Custom CNN Architecture**

- **Description:** Model A is designed using a custom Convolutional Neural Network (CNN) architecture tailored specifically for classifying images as either AI-generated or real. The network consists of two convolutional layers with ReLU activation functions followed by max pooling layers, enabling it to effectively extract visual features from the input images. The fully connected layers at the end transform the feature maps into class predictions for binary classification. 

- **Initialization:** In this model, all layers are initialized from scratch, which allows for a focused learning process tailored to the specific dataset. The architecture employs standard initialization techniques for convolutional and fully connected layers, which helps in achieving stable training.

- **Benefits:** The tailored architecture and training strategy enable Model A to effectively learn the distinguishing features between AI-generated and real images, resulting in high accuracy and strong generalization capabilities without signs of overfitting.


## 2. **Model B: Transfer Learning**

- **Description:** This model utilizes a combination of advanced architectures, including *ResNet, EfficientNet, and Vision Transformer (ViT)*, to enhance performance in classifying AI-generated versus real images. Each model is chosen for its unique strengths: 
   - **ResNet** effectively addresses vanishing gradient issues through its skip connections, facilitating the training of deeper networks.
   - **EfficientNet** optimizes model size and accuracy through a compound scaling method, allowing for better performance with fewer resources.
   - **Vision Transformer (ViT)** excels in capturing long-range dependencies in image data by processing images as sequences of patches, leveraging self-attention mechanisms.

- **Initialization:** This model employs a similar CNN architecture to Model A, but adopts a distinct initialization strategy. Specifically, some layers are initialized with weights from pre-trained models, while others are randomly initialized. This technique involves freezing select layers from the pre-trained models, enabling the model to leverage established knowledge while effectively adapting to our specific task.

- **Benefits:** By incorporating transfer learning across these diverse architectures, this model accelerates training and enhances performance, particularly when working with smaller datasets. It capitalizes on the strengths of proven architectures, allowing for robust feature extraction and improved generalization capabilities while adapting them to our unique requirements.


## 3. **Model C: Statistical Model**

- **Description:** This approach utilizes traditional machine learning models to analyze statistical features extracted from the images. Key characteristics such as pixel intensity distributions, texture, and edge structures are examined to create a model that effectively distinguishes between real and AI-generated faces.

- **Preparation & Initialization:** Key steps for preparing and initializing the statistical models include extracting relevant statistical features from images, such as Fourier features and gradient statistics, to create a robust dataset. The training data is properly prepared by splitting, normalizing, and addressing any imbalances. Hyperparameters for the chosen models, such as those in Random Forest or Gradient Boosting, are set for optimal performance. Additionally, establishing a random seed can enhance reproducibility. Proper initialization is crucial for ensuring that the models effectively capture the underlying patterns in the data.

- **Benefits:** The statistical model provides a complementary perspective to deep learning approaches, offering insights into the underlying features that contribute to classification. It also serves as a baseline for evaluating the performance of more complex models.

# 📊 Summary of Extracted Statistical Features
### Feature Computation
- **Fourier Features (fourier_features(gray_img))**: A list of features derived from the 2D Fourier transform, including the radial power spectrum, log power slope, deviation from the natural 1/f trend, and band power ratios.
- **Radial Power Profile (radial_power_profile(log_mag))**: Normalized radial averages of the log magnitude spectrum, providing insights into frequency distribution across images.
- **Gradient Statistics (gradient_stats(gray_img))**: Mean edge strength, standard deviation, skewness, and kurtosis, characterizing the edge content of a grayscale image.
- **Local Contrast Statistics (local_contrast_stats(gray_img))**: Key statistics for local variance using a blurring technique, aiding in the evaluation of local texture and contrast in grayscale images.
- **Residual Noise Statistics (residual_noise_stats(gray_img))**: Noise statistics and power from the residuals after Gaussian blurring, evaluating noise levels in images.
- **Global Intensity Statistics (global_intensity_stats(gray_img))**: Global statistics of brightness and contrast, enabling the evaluation of overall image intensity distribution and pixel value characteristics.

### Overall Importance Ranking 

<img width="726" height="401" alt="Screenshot 2025-12-05 at 12 19 03 PM" src="https://github.com/user-attachments/assets/7adb0b6f-6cc3-42ff-ba1d-7d8930a7596c" />


# 📈 Evaluation Across All Five Models
Each model exhibits strong performance, achieving high accuracy on both validation and testing datasets. The classification reports reveal excellent F1-Scores for both AI-generated and real images, underscoring the effectiveness of all models in tackling the classification task. Overall, they all excel in distinguishing between AI-generated and real images.

<img width="686" height="376" alt="Model_performance_comparison" src="https://github.com/user-attachments/assets/35b5859a-879a-49b0-b509-96852735035e" />

### Comparison of Learning Curves
In the folder *'results'*, you will find the learning curves for four models: PyTorch CNN, EfficientNet, ResNet, and Vision Transformer. Each model demonstrates strong learning capabilities and high accuracy in classifying AI-generated versus real images. EfficientNet and ResNet excel in generalization with minimal signs of overfitting, while the Vision Transformer shows potential but may benefit from further tuning to enhance test accuracy stability. Overall, these results highlight the effectiveness of different architectures in tackling the classification task, providing valuable insights for future model selection and refinement.

Note that statistical models lack learning curves because they estimate parameters directly from the entire dataset rather than through iterative training. They produce a single set of results without multiple epochs, focusing more on interpretability and relationships between variables than on continuous performance tracking.

### Visualization of Predictions
<img width="1478" height="591" alt="Screenshot 2025-12-03 at 10 46 24 AM" src="https://github.com/user-attachments/assets/ff1e32fd-6891-4a26-9277-e02fb944af91" />


# 🔍 Comparison of the three approaches
- **Execution Time:** Time taken for execution canvary significantly; however, it is challenging to determine precise execution times as I utilize a shared GPU on Google Colab. In this context, machine learning models may take longer due to GPU processing, while Model 3 (the statistical image model) does not require a GPU, making it inherently faster to execute.
- **Applicability in Real-World Situations:**   
   - Machine learning models often excel in handling complex datasets and can adapt to various scenarios, making them suitable for diverse applications. However, their dependency on substantial computational resources may limit accessibility.
   - In contrast, the statistical image model is simpler and more interpretable, making it easier to implement in real-world situations where speed and resource constraints are critical, especially since it does not require GPU resources.
 
# 🌟 Conclusion and Future Work
In this project, I employed a diverse set of models—including Custom CNN, ResNet-18, EfficientNet-B0, and Vision Transformer (ViT) B-16, alongside various statistical methods to achieve robust performance in classifying AI-generated versus real images. Each model's unique architecture was carefully selected to address the specific challenges of this task, maximizing the strengths of both deep learning and statistical analysis. While the initial results appear promising, I have recognized that the performance reported by the models may not reflect realistic capabilities. The dataset comprises only 10,000 images (5,000 real and 5,000 AI-generated) raising concerns about the models' generalizability to new, unseen data. The limited diversity, sourced from just four channels, may not capture the vast variability present in real-world scenarios. Variations in lighting, backgrounds, and subjects can significantly influence performance, leading to inflated metrics. Additionally, the risk of overfitting is heightened due to the small dataset size. The models may memorize training data rather than identify underlying patterns, which could result in excellent performance on training and validation sets but struggles with new data. Therefore, it is crucial to monitor performance closely for signs of overfitting. To address these challenges, I plan to implement several strategies:

- *Cross-Validation*: This technique can provide a more accurate assessment of model performance by evaluating how well each model performs on different subsets of the data.
- *Data Augmentation*: Techniques such as rotation, scaling, and flipping can artificially increase dataset diversity, potentially improving generalization.
- *External Datasets*: Testing the models on datasets not encountered during training can yield valuable insights into their real-world performance, ensuring they recognize patterns across a broader spectrum of images.

In conclusion, while the initial results are encouraging, addressing the limitations posed by the dataset's size and diversity is essential. By implementing robust validation techniques and enhancing the dataset, I aim to develop models that perform effectively in real-world applications. 🌍

