---
layout: post
title: Create Your Own Neural Network from Scratch
date: 2024-02-14 10:00:00
description: How to make your own neural network using Python from scratch
tags: formatting code
categories: machine-learning
featured: true
---

# Create Your Own Neural Network for Dummies
In this blog post, we'll explore how to create a simple neural network from scratch using Python. We'll walk through building a Single Layer Perceptron to classify diabetes patients based on a dataset from Kaggle. You don’t need any fancy deep learning libraries like TensorFlow or PyTorch; we'll implement everything from scratch. By the end of this blog, you'll understand the core concepts behind neural networks and have a functional perceptron model. Let's get started!

---

## What is a Single Layer Perceptron?

A **Single Layer Perceptron (SLP)** is the simplest type of artificial neural network. It consists of a single layer of weights that connect the input features to the output node, with an activation function applied to produce the final output. The perceptron was originally introduced by Frank Rosenblatt in 1958 as a binary classifier—its job is to determine whether an input belongs to one of two classes.
### How Does It Work?

In a Single Layer Perceptron:

1. **Inputs (Features)**: The perceptron takes multiple input features (e.g., attributes from the dataset).
2. **Weights**: Each input is associated with a weight that determines its contribution to the output. These weights are initially set randomly and updated during the training process.
3. **Weighted Sum Calculation**: The perceptron computes the weighted sum of the inputs and biases.
4. **Activation Function**: The weighted sum is then passed through an activation function that maps the output to a desired range, usually between 0 and 1 for binary classification.
5. **Output**: The activation function's output is used to make the final prediction.
### Understanding a Single Layer Perceptron

In this diagram:


```
       Input Layer                      Output
   +-------------+                 +------------+
   |             |   Weights       |            |
   |   (X1)  ● ---- W1 ----+       |     ●      |
   |             |         |       |   Output   |
   |   (X2)  ● ---- W2 ----|-----> |     (Y)    |
   |             |         |       |            |
   |   (X3)  ● ---- W3 ----+       +------------+
   |             |
   +-------------+
```

- Each input feature (`X1`, `X2`, `X3`) is represented as a circle in the input layer.
- These inputs are connected to the output neuron through weighted connections (`W1`, `W2`, `W3`).
- The output neuron computes a weighted sum of the inputs and applies an activation function to produce the output (`Y`).

### Implementing the Custom Perceptron Network

####  Setting Up the Diabetes Dataset

To train our perceptron, we'll use the **Diabetes Dataset** from Kaggle. This dataset contains several medical attributes such as glucose levels, BMI, and insulin levels, which are used to classify whether a patient is diabetic (1) or non-diabetic (0). The target variable is outcome which is in binary format (0 & 1)

Link: [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)
### Data Preparation

First, let's load and prepare the data:

**Normalize the Features**: It's important to scale the input features so that the model can learn effectively. Each feature is in a different measure so we will be using StandardScaler to normalize the values.

There are 8 Features in the dataset

    - Pregnancies: Number of times pregnant
	- Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
	- BloodPressure: Diastolic blood pressure (mm Hg)
	- SkinThickness: Triceps skin fold thickness (mm)
	- Insulin: 2-Hour serum insulin (mu U/ml)
	- BMI: Body mass index (weight in kg/(height in m)^2)
	- DiabetesPedigreeFunction: Diabetes pedigree function
	- Age: Age (years)
	
**Train-Test Split** : We’ll split the dataset into training and testing sets for model evaluation

**Importing the data and Splitting it into Train and Test sets**

```
import pandas as pd
data = pd.read_csv('diabetes.csv')

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values # The last feature is the target variable "Outcome"

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

```

**Applying Normalization using Standard Scaler from sklearn**

```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Building the Perceptron from Scratch

In this section, we will walk through the implementation of a Single Layer Perceptron (SLP) using Python. We'll break down the key components of the code and explain how each part contributes to the training and prediction process. We initialize a function called train_neural_net and this willl have four parameters X,Y,learning_rate,epochs.
X - the features
Y- the target variable
learning rate - This determines how much the weights are updated at each step. 
Epoch - Iterating on the each input and modifying the weights each time.
#### Initializing the Weights

We will generate weights for each feature in X_train using np.random.rand function and converting the weights to a numerical list (since the np library generates all values in numpy format)

```
weights = np.random.rand(X_train.shape[1])
w = weights.tolist() 
```

#### Running each input into the Network

```
for epoch in range(epochs):
        ypred = []
        for i in range(len(X)):
            y = sum(X[i][j] * w[j] for j in range(len(w)))
            output = 1 if y > 0.5 else 0
            ypred.append(output)
```

#### Writing a simple Binary Activation function

```
			output = 1 if y > 0.5 else 0
```

#### Updating the error using Gradient Descent

```
			error = Y[i] - output

			for j in range(len(w)):
                w[j] += learning_rate * error * X[i][j]
```

#### Final Neural Network function

```
def train_neural_net(X, Y, learning_rate=0.01, epochs=100):
    # Initialize Weights randomly
    weights = np.random.rand(X.shape[1])
    w = weights.tolist()
    for epoch in range(epochs):
        ypred = []
        for i in range(len(X)):
            y = sum(X[i][j] * w[j] for j in range(len(w)))
            # Activation Function is applied in the output variable
            output = 1 if y > 0.2 else 0
            ypred.append(output)
            error = Y[i] - output
            # Updating the error using Gradient Descent
            for j in range(len(w)):
                w[j] += learning_rate * error * X[i][j]
    return w
```

#### Prediction function

```
def predict(X, w):
    ypred = []
    for i in range(len(X)):
        y = sum(X[i][j] * w[j] for j in range(len(w)))
        # Apply the binary activation function
        output = 1 if y > 0.2 else 0  # Using a threshold of 0.5
        ypred.append(output)
    return ypred
```

#### Calling the function and Applying it to our dataset

We will now send the test set to the predict function with the weights from our train_neural_net 

```
final_weights = train_neural_net(X_train, Y_train)
ypred=predict(X_test,final_weights)
```

#### Evaluating the Model Performance

We will be using the metrics module from sklearn to implement the evaluation of our model using accuracy_score, confusion_matrix and classification_report. This helps us to understand How our model precisely identifies the diabetic patients

```
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(Y_test,ypred)
conf_matrix = confusion_matrix(Y_test, ypred)
class_report = classification_report(Y_test, ypred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
```

#### Result

```
Accuracy: 76.62%

Confusion Matrix:
[[89 10]
 [26 29]]

Classification Report:
              precision    recall  f1-score   support

           0       0.77      0.90      0.83        99
           1       0.74      0.53      0.62        55

    accuracy                           0.77       154
   macro avg       0.76      0.71      0.72       154
weighted avg       0.76      0.77      0.76       154
```

The model achieved an accuracy of 76.62%, indicating it performed reasonably well in predicting diabetes but showed some limitations, especially in identifying diabetic cases (Class 1). 

The precision and recall for Class 1 suggest that while the model identified many true positives, it missed a significant portion of them, leading to a lower recall. 

To improve the model, we could try techniques like hyperparameter tuning, increasing the number of neurons or layers in the network, using a balanced dataset with techniques like SMOTE to address class imbalance, and experimenting with advanced optimizers or regularization methods to enhance generalization.
