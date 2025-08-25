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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import torch

data=pd.read_csv('/content/linear.csv')
x=data.iloc[:,0].values
y=data.iloc[:,1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train.reshape(-1, 1))
x_test=scaler.transform(x_test.reshape(-1, 1))

x_train_tensor=torch.tensor(x_train,dtype=torch.float32)
y_train_tensor=torch.tensor(y_train,dtype=torch.float32).view(-1,1)
x_test_tensor=torch.tensor(x_test,dtype=torch.float32)
y_test_tensor=torch.tensor(y_test,dtype=torch.float32).view(-1,1)

class NeuralNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=torch.nn.Linear(1,8)
    self.fc2=torch.nn.Linear(8,12)
    self.fc3=torch.nn.Linear(12,1)
    self.relu=torch.nn.ReLU()
    self.history={'loss':[]}

  def forward(self,x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.fc3(x)
    return x

cynthia_brain=NeuralNetwork()
loss_fn=torch.nn.MSELoss()
optimizer=torch.optim.RMSprop(cynthia_brain.parameters(),lr=0.001)

def train_model(model, x_train, y_train, loss_fn, optimizer, epochs):
  for epoch in range(epochs):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(cynthia_brain, x_train_tensor, y_train_tensor, loss_fn, optimizer,1500)

with torch.no_grad():
  y_pred = cynthia_brain(x_test_tensor)
  test_loss = loss_fn(y_pred, y_test_tensor)
  print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(cynthia_brain.history)
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[23]], dtype=torch.float32)
prediction = cynthia_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
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
