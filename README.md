# EXPERIMENT 01: DEVELOPING A NEURAL NETWORK REGRESSION MODEL
## AIM:
To develop a neural network regression model for the given dataset.

## THEORY:
Neural network regression models learn complex relationships between input variables and continuous outputs through interconnected layers of neurons. By iteratively adjusting parameters via forward and backpropagation, they minimize prediction errors. Their effectiveness hinges on architecture design, regularization, and hyperparameter tuning to prevent overfitting and optimize performance.
### Architecture:
  This neural network architecture comprises two hidden layers with ReLU activation functions, each having 5 and 3 neurons respectively, followed by a linear output layer with 1 neuron. The input shape is a single variable, and the network aims to learn and predict continuous outputs.

## NEURAL NETWORK MODEL:
![image](https://github.com/Rithigasri/basic-nn-model/assets/93427256/de6017e4-fcd7-4a31-abbd-9a36bd7ae689)

## DESIGN STEPS:
### STEP 1:
Loading the dataset.
### STEP 2:
Split the dataset into training and testing.
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
## PROGRAM:
### Name: chowla chaithanya
### Register Number: 2305002004
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('DL').sheet1

rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df=df.astype({'INPUT':'float'})
df=df.astype({'OUTPUT':'float'})
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = df[['INPUT']].values
y = df[['OUTPUT']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

model=Sequential([
    #Hidden ReLU Layers
    Dense(units=5,activation='relu',input_shape=[1]),
    Dense(units=3,activation='relu'),
    #Linear Output Layer
    Dense(units=1)
])

model.compile(optimizer='rmsprop',loss='mse')
model.fit(X_train1,y_train,epochs=3000)

loss= pd.DataFrame(model.history.history)
loss.plot()

X_test1 =Scaler.transform(X_test)
model.evaluate(X_test1,y_test)

X_n1=[[4]]
X_n1_1=Scaler.transform(X_n1)
model.predict(X_n1_1)

```
## DATASET INFORMATION:
<img width="1039" height="726" alt="image" src="https://github.com/user-attachments/assets/77601fe1-3ca0-43bf-a5a0-13407ccd7a89" />



## OUTPUT:
### Training Loss Vs Iteration Plot:
<img width="951" height="681" alt="image" src="https://github.com/user-attachments/assets/fa3ba3f6-42cf-49de-a60f-896298de0cd7" />

### Epoch Training:
<img width="926" height="282" alt="image" src="https://github.com/user-attachments/assets/a2578947-ae88-4c1c-b13e-68472a6e6dae" />

### Test Data Root Mean Squared Error:
<img width="916" height="114" alt="image" src="https://github.com/user-attachments/assets/4069fc3e-0f80-483f-8379-2894e38fcf89" />

### New Sample Data Prediction:
<img width="813" height="97" alt="image" src="https://github.com/user-attachments/assets/c2ddfca2-0c7f-421f-9bb6-d22da5342080" />



## RESULT:
Thus a basic neural network regression model for the given dataset is written and executed successfully.
