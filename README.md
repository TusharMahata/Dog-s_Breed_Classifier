# Dog-s_Breed_Classifier
Dataset

The Stanford Dogs dataset comprises 20,580 images of 120 breeds of dogs sourced from ImageNet. Originally collected for fine-grained image categorization, the dataset poses challenges due to similar features among certain breeds. The original data source provides additional details on train/test splits and baseline results.
Preprocessing

    Load the dataset and extract class names from the folder URLs.
    Visualize the number of pictures for each dog's breed.
    Implement a TensorFlow data pipeline for efficient image classification.

Model_0

    Base model layer created using MobileNetV3.
    Simple model built and trained with early stopping.

Model_0 Output

The initial result showed signs of overfitting.
Augmentation Implementation

To address overfitting and enhance results:

    Add a function, process_augment_image, to the data pipeline for image augmentation.
    Retrain the model.

Model_1

    A neural network model (model_1) designed using TensorFlow's Keras API.
    It includes a base model, two dense hidden layers with ReLU activation and dropout layers, and an output layer with softmax activation.
    Compiled with categorical crossentropy loss, Adam optimizer, and accuracy metric.
    Built and trained on augmented data (train_data_aug) with early stopping and model checkpointing.

API

    FastAPI application for Dog's Breed Classification using a pre-trained model (my_model).
    Endpoints:
        Root ("/"): Welcome message.
        "/predict": Handles image uploads, processes images, makes predictions, and returns predicted class.

Docker

Dockerized the FastAPI project for portability and reproducibility:

    Created a Dockerfile specifying the environment and dependencies.
    Listed required packages in requirements.txt.
    Built the Docker image and ran a container.
    Accessed the API at http://localhost:6565.

Feel free to explore and contribute to this Dog's Breed Classifier project!


