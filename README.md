# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Developing a neural network model to perform regression tasks efficiently using the given dataset. The neural network has 1 neuron in the input layer, 4 neurons in the first hidden layer, 8 neurons in the second hidden layer and 1 neuron in the output layer. RMSprop is used as the optimizer for weight adjustment and MSE is used to compute the loss at each epoch.

## Neural Network Model

<img width="1031" height="793" alt="image" src="https://github.com/user-attachments/assets/4fa8365b-ef40-4a76-a97c-1d837a79f838" />

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
### Name: NATARAJ KUMARAN S
### Register Number: 212223230137

```python
import torch.nn as nn
class Neural(nn.Module):
  def __init__(self):
    super().__init__()
    # Defines the layers based on the image's architecture
    self.fc1=nn.Linear(1, 8)    # Input (1) -> Hidden Layer 1 (8)
    self.fc2=nn.Linear(8, 12)   # Hidden Layer 1 (8) -> Hidden Layer 2 (12)
    self.fc3=nn.Linear(12, 1)   # Hidden Layer 2 (12) -> Output (1)
    self.relu=nn.ReLU()
    self.history={'loss':[]}

  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x
```
```python
ai = Neural()
criterion=nn.MSELoss()
optimizer=torch.optim.RMSprop(ai.parameters(),lr=0.001)
```
```python
def train_model(ai,X_train,y_train,criterion,optimizer,epochs=3000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(ai(X_train),y_train)
    loss.backward()
    optimizer.step()

    ai.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

```
## Dataset Information

<img width="462" height="610" alt="image" src="https://github.com/user-attachments/assets/c648ec0f-7063-4220-8d3f-ad719d2337d7" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="747" height="568" alt="image" src="https://github.com/user-attachments/assets/9d814f1a-8872-4c5c-9af9-191c3713de03" />

### New Sample Data Prediction


<img width="1548" height="147" alt="image" src="https://github.com/user-attachments/assets/9f171916-677a-40b2-911b-da487c7bc94d" />



## RESULT
Therefore, neural network is developed for the basic dataset and implemented successfully. 
