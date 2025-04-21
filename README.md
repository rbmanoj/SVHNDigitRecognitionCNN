# Street View Housing Number Digit Recognition using Deep Learning

## Background

This project focuses on recognizing digits in natural scene images using deep learning, specifically the Street View Housing Number (SVHN) dataset. SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the same number of classes), but incorporates an order of magnitude more labeled data (over 600,000 digit images in total) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images.

## Dataset

The project utilizes a subset of the original SVHN dataset to reduce computational time. The dataset is provided as an HDF5 (.h5) file containing the following:

*   **X_train:** Training data consisting of 42,000 grayscale images of digits, each a 32x32 pixel matrix.
*   **y_train:** Corresponding labels for the training images, representing digits 0-9.
*   **X_test:** Test data consisting of 18,000 grayscale images of digits, each a 32x32 pixel matrix.
*   **y_test:** Corresponding labels for the test images, representing digits 0-9.


## Approach

The project employs two main approaches for digit recognition:

1. **Artificial Neural Networks (ANNs):** Two ANN models with varying architectures are built and trained. The models use fully connected layers with ReLU activation functions, dropout for regularization, and batch normalization.

2. **Convolutional Neural Networks (CNNs):** Two CNN models are developed, leveraging convolutional layers with LeakyReLU activations, max-pooling for downsampling, and batch normalization. Dropout is also incorporated for regularization.

Both approaches involve data preprocessing steps such as normalization and one-hot encoding of the target variables. Model performance is evaluated using metrics like accuracy, precision, recall, and F1-score. Confusion matrices are also used to visualize the model's predictions and identify areas of improvement.

The project aims to compare the performance of ANNs and CNNs on the SVHN dataset and identify the best-performing model for street view housing number digit recognition.
