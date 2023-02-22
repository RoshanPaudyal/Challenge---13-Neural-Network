# Challenge-13-Neural-Network

### README for AlphabetSoup Charity Deep Learning Model
#### Overview
The purpose of this project is to develop and optimize a deep learning model to predict whether applicants will be successful if funded by Alphabet Soup, a philanthropic foundation that provides financial support for organizations that promote education, literacy, and programs that help people learn to read and write.

#### Dependencies
This project was written in Python 3.8.8, using the following packages:

1. pandas==1.3.3
2. scikit-learn==1.0
3. tensorflow==2.6.0
 
#### Getting Started
To run the code in this project, follow these steps:

1. Clone the repository to your local machine
2. Navigate to the project directory in your terminal
3. Install the dependencies using pip install -r requirements.txt
4. Run the Jupyter Notebook using jupyter notebook in your terminal
 
#### Data
The dataset used in this project is a CSV file containing more than 34,000 organizations that have received funding from Alphabet Soup. The data includes information such as the organization's name, the type of application, the amount of funding requested, and the success of the application.

####  Model Development

##### Original Model
The original deep learning model was developed with an input layer, two hidden layers, and an output layer. The model used the relu activation function for the hidden layers and the sigmoid function for the output layer.
For the original model, we followed these steps:

1. Data Cleaning: We loaded the charity_data.csv dataset and preprocessed the data by dropping unnecessary columns and encoding categorical data.
2. Data Preprocessing: We split the preprocessed data into features (X) and target (y) arrays, performed a train-test split, and scaled the data using StandardScaler.
3. Model Creation: We created a neural network model with three hidden layers using the Keras library. We used the "relu" activation function in the hidden layers and "sigmoid" in the output layer.
4. Model Compilation and Fitting: We compiled the model with the "binary_crossentropy" loss function, "adam" optimizer, and "accuracy" metrics. Then, we fit the model using 50 epochs and the training data.
4. Model Evaluation: We evaluated the model's loss and accuracy metrics using the evaluate method and the test data.

##### Alternative Model 1
1. Load the AlphabetSoupCharity data.
2. Preprocess the data by performing the following steps:.
- Encode categorical variables using one-hot encoding.
- Standardize numerical variables using the StandardScaler module from Scikit-learn.
- Merge the preprocessed data into a single dataset.
- Split the preprocessed data into training and testing datasets.
7. Define the architecture of the alternative model by setting the number of input features, hidden nodes for the first hidden layer, and number of neurons in the output layer.
8. Create the alternative model using the TensorFlow Keras library.
9. Compile the alternative model by specifying the loss function, optimizer, and accuracy metric.
10. Train the alternative model using the training dataset and set the number of epochs.
11. Evaluate the performance of the alternative model using the test dataset.

##### Alternative Model 2
1. Load the AlphabetSoupCharity data.
2. Preprocess the data by performing the following steps:
- Encode categorical variables using one-hot encoding.
- Standardize numerical variables using the StandardScaler module from Scikit-learn.
- Merge the preprocessed data into a single dataset.
- Split the preprocessed data into training and testing datasets.
3. Define the architecture of the alternative model by setting the number of input features, hidden nodes for the first hidden layer, and number of neurons in the output layer.
4. Create the alternative model using the TensorFlow Keras library.
5. Compile the alternative model by specifying the loss function, optimizer, and accuracy metric.
6. Train the alternative model using the training dataset and set the number of epochs.
7. Evaluate the performance of the alternative model using the test dataset.


#### Results
1. Original Model Results
- Loss: 0.5440145735740662
- Accuracy: 0.7265560989379883

2. Alternative Model 1 Results
- Loss: 0.556940197467804
- Accuracy: 0.7252790927886963

3. Alternative Model 2 Results
- Loss: 0.588522970199585
- Accuracy: 0.7242029304504395

#### Conclusion
From these results, we can see that all three models have very similar accuracies, with the original model and Alternative Model 2 having slightly higher accuracy than Alternative Model 1. However, the differences in accuracy between the models are quite small. The loss values are also quite similar between the models. This suggests that there may not be much to gain in terms of predictive accuracy by continuing to tweak the neural network model.
