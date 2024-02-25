# Developing a Neural Network Regression Model

## AIM

#### To develop a neural network regression model for the given dataset.

## THEORY

##### A neural network is a computer program inspired by how our brains work. It's used to solve problems by finding patterns in data. Imagine a network of interconnected virtual "neurons." Each neuron takes in information, processes it, and passes it along.

##### A Neural Network Regression Model is a type of machine learning algorithm that is designed to predict continuous numeric values based on input data. It utilizes layers of interconnected nodes, or neurons, to learn complex patterns in the data. The architecture typically consists of an input layer, one or more hidden layers with activation functions, and an output layer that produces the regression predictions.

##### This model can capture intricate relationships within data, making it suitable for tasks such as predicting prices, quantities, or any other continuous numerical outputs.

## Neural Network Model

![image](https://github.com/PSriVarshan/basic-nn-model/assets/114944059/4b81f952-bf11-4e78-b7f8-a1b79090ed09)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: SANJAY T
### Register Number: 212222110039
python

# Importing Modules:

```from google.colab import auth
import gspread
from google.auth import default

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den

from tensorflow.keras.metrics import RootMeanSquaredError as rmse

import pandas as pd
import matplotlib.pyplot as plt
```
# Authenticate & Create Dataframe using Data in Sheets:
```
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

sheet = gc.open('Mysample').sheet1 
rows = sheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
```
# Assign X and Y values:
```
x = df[["Input"]] .values
y = df[["Output"]].values
```
# Normalize the values & Split the data:
```
scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)
```
# Create a Neural Network & Train it:
```
ai_brain = Seq([
    Den(9,activation = 'relu',input_shape=[1]),
    Den(16,activation = 'relu'),
    Den(1),
])

ai_brain.compile(optimizer = 'rmsprop',loss = 'mse')

ai_brain.fit(x_train,y_train,epochs=1000)
ai_brain.fit(x_train,y_train,epochs=1000)

# Plot the Loss:

loss_plot = pd.DataFrame(ai_brain.history.history)
loss_plot.plot()
```
# Evaluate the model:
```
err = rmse()
preds = ai_brain.predict(x_test)
err(y_test,preds)
```
# Predict for some value:
```
x_n1 = [[9]]
x_n_n = scaler.transform(x_n1)
ai_brain.predict(x_n_n)

```
## Dataset Information

![image](https://github.com/PSriVarshan/basic-nn-model/assets/114944059/ce55eff8-d273-45a6-88a5-75a315e7859e)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/PSriVarshan/basic-nn-model/assets/114944059/9ee62e31-4b9b-4c3d-9930-d4355427add3)


### Test Data Root Mean Squared Error

![image](https://github.com/PSriVarshan/basic-nn-model/assets/114944059/e5e1aadd-8289-4a77-8e4c-7a9d013fc7cf)


### New Sample Data Prediction

![image](https://github.com/PSriVarshan/basic-nn-model/assets/114944059/25faf495-8cf9-44a9-b189-09c8f8f7dfb2)

## RESULT

### Thus to develop a neural network regression model for the dataset created is successfully executed.
