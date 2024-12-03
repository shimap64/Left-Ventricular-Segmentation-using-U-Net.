Overview
This project implements a U-Net-based architecture for the segmentation of the left ventricle in medical images.
U-Net is a popular convolutional neural network designed for biomedical image segmentation tasks. 
The implementation includes data augmentation, model training, evaluation metrics, and visualization of results.

Steps in the Code:
1. Environment Setup
TensorFlow is used as the primary library for building and training the U-Net model.
The availability of GPU is checked for optimized performance.
2. Data Preprocessing
Loading and Augmentation:
A function applies random affine transformations (scaling, rotation, flipping, and translation).
Noise is optionally added to the images for robustness.
Dataset Preparation:
Data is split into training and validation sets.
Resizing ensures images and masks are compatible with the model.
3. U-Net Model
A U-Net model is defined with an encoder-decoder architecture.
Loss and Metrics:
Dice Coefficient: Measures overlap between ground truth and predicted masks.
Intersection over Union (IoU): Quantifies segmentation accuracy.
Precision, Recall, and F1-Score: Provide insights into the performance for true positives, false positives, and false negatives.
4. Model Training
The model is compiled with appropriate loss functions and metrics.
Training is performed using the prepared datasets, with evaluation on the validation set.
5. Predictions and Evaluation
The trained model is used to predict masks on validation images.
The segmentation results are evaluated using various metrics and visualized alongside ground truth masks.

Materials and Images:
1. U-Net Architecture
The U-Net architecture comprises a contracting path (encoder) and an expansive path (decoder).
It captures spatial information and context for precise localization.

![image](https://github.com/user-attachments/assets/f6175536-489c-4cee-8190-6d48565ca38a)

2. Data Augmentation
Example of augmented images:
•	Image: Original
•	Mask: Transformed mask


Augmentation Type	Effect
Flip	Random flipping of images and masks.
Scale	Adjusts size within a defined range.
Rotation	Random rotation (±10°).
Noise Addition	Gaussian noise for robustness.

Summary of the Model Architecture
The presented model is a functional neural network architecture designed for image data processing, likely a U-Net variant based on its encoder-decoder structure. Below are the highlights:

Input Layer: Accepts input images of shape (256, 256, 3).

Encoder (Downsampling Path):

Composed of stacked Conv2D layers with increasing filters (64, 128, 256, 512, 1024) followed by BatchNormalization, Dropout, and MaxPooling2D layers.
Sequential feature extraction and size reduction are performed.
Bottleneck: The deepest layer in the network with 1024 filters. This section contains the most abstract feature representations.

Decoder (Upsampling Path):

Conv2DTranspose layers are used for upsampling, paired with concatenation layers to reuse features from the encoder (skip connections).
Each upsampling step is followed by Conv2D, BatchNormalization, and Dropout layers.
Output Layer: Designed to provide predictions, typically with a final activation function suitable for the task (e.g., softmax for multi-class segmentation or sigmoid for binary segmentation).

Skip Connections: These connections between encoder and decoder layers help retain spatial information and improve gradient flow during backpropagation.

Parameters: The network consists of tens of millions of parameters due to its depth and large filter sizes.



3. Results Table
Metric	Value
Dice Coefficient	0.92
IoU Score	0.85
Precision	0.91
Recall	0.93
F1-Score	0.92
Summary of the Model Training Results:
Key Metrics Across 10 Epochs
Training Metrics:

Dice Coefficient: Steadily improved, reaching 0.8903 by the 10th epoch.
F1-Score Metric: Parallels Dice Coefficient, ending at 0.8903.
IoU (Intersection over Union): Gradually increased to 0.8023 by the final epoch.
Loss: Decreased significantly, reaching 0.0392 by epoch 10.
Precision: Ended at 0.8905.
Recall: Consistent improvement, reaching 0.8905.
Validation Metrics:

Dice Coefficient: Peaked at 0.8909 in the final epoch, showing consistent improvement.
F1-Score Metric: Matches the Dice Coefficient at 0.8909.
IoU: Improved throughout training, ending at 0.8033.
Loss: Declined, reaching 0.0341 by epoch 10.
Precision: Peaked at 0.9022, indicating strong predictive confidence.
Recall: Consistently high, ending at 0.8800.

predictions 

![image](https://github.com/user-attachments/assets/8d38ab7b-3548-474e-903b-1c6a22a681b6)
![image](https://github.com/user-attachments/assets/5d4b2c48-869a-4cd2-ba0b-26e3b35ad715)
![image](https://github.com/user-attachments/assets/df8241ab-6b7a-4070-a9e7-93cd1848c234)




