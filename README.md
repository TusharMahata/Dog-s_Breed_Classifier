# Dog-s_Breed_Classifier
Dataset:
The Stanford Dogs dataset contains a total 20,580 images of 120 breeds of dogs from around the world. This dataset has been built using images and annotation from ImageNet for the task of fine-grained image categorization. It was originally collected for fine-grain image categorization, a challenging problem as certain dog breeds have near identical features or differ in color and age.
The original data source is found on http://vision.stanford.edu/aditya86/ImageNetDogs/ and contains additional information on the train/test splits and baseline results.


Preprocessing: 
Load the dataset and split class names from the folder url. 
Visualization of number of pictures for each dog's breed of dataset. 
TensorFlow data pipeline for image classification. It includes functions for retrieving image data, processing images, and configuring TensorFlow Datasets. The main function, image_data_pipeline, can create batches for either test data or training/validation data based on specified parameters such as image size and batch size. This modular script is designed for efficient image handling in machine learning tasks.
Model_0 : 
Creating the base model layer by mobilenet_v3.
Build a simple model and train using early_stopping callback. 

Output result seems a bit overfitting. To prevent and improve the result-
Add a function in image_data_pipeline called process_augment_image to do image augmentation.
Train the model. 

Model Summary: A neural network model (model_1) using TensorFlow's Keras API. It consists of a base model, two dense hidden layers with ReLU activation and dropout layers, and an output layer with softmax activation. The model is compiled with categorical crossentropy loss, the Adam optimizer, and accuracy as the metric. It is then built and trained on augmented data (train_data_aug) with early stopping and model checkpointing. 

API:
FastAPI application for a Dog's Breed Classifier using a pre-trained model. Here's a breakdown:
Loading the Model:
A pre-trained Keras model (my_model) is loaded using tf.keras.models.load_model.
Class Names:
A list of class names (class_names) corresponding to dog breeds is defined.
Model Prediction Pipeline:
The model_pipeline function takes an image, normalizes it, converts it to a TensorFlow tensor, predicts the class using the loaded model, and returns the predicted class name.
FastAPI Setup:
A FastAPI instance (app) is created.
A root endpoint ("/") returns a welcome message.
The load_image_into_numpy_array function converts the uploaded image into a NumPy array.
The "/predict" endpoint handles image uploads, processes the image, makes predictions using the model, and returns the predicted class.
Endpoint to Make Predictions:
The /predict endpoint accepts a file upload (file: UploadFile).
The uploaded image is processed using the load_image_into_numpy_array function.
The model_pipeline function is called to make predictions.
The predicted class is returned as the API response.


Docker: 
Dockerizing the FastAPI project involves creating a Dockerfile and a requirements.txt file. The Dockerfile specifies the environment and dependencies, and the requirements.txt lists necessary packages. Build the Docker image and run a container, making your FastAPI app portable and reproducible across different environments. Access the API at http://localhost:6565. 


