# Fashion-MNIST computer vision classification   

In this notebook, we will build a convolutional neural network (CNN) model using TensorFlow and Keras to classify images from the Fashion-MNIST dataset. The Fashion-MNIST dataset is a collection of images of clothing items such as T-shirts, dresses, and shoes. The goal is to create a model that can accurately classify images into their corresponding categories.

This work has been made using the following resources:
- Previous work on MNIST numbers classification
- GeekForGeeks documentation
- TensorFlow documentation
- ChatGPT to give him my errors
## Importing libraries

We begin by importing the necessary libraries. We import `fetch_openml` from `sklearn.datasets` to download the Fashion-MNIST dataset. We also import `train_test_split` to split the dataset into training and testing sets. `matplotlib.pyplot` is used to display the images. Finally, we import `tensorflow.keras` to build and train our model.

## Loading and Preprocessing Data  
We then load the Fashion-MNIST dataset using `fetch_openml`. We set `n_samples` to 4000 to limit the dataset size. We split the dataset into training and testing sets using `train_test_split`.  

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
# Define the category names
categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# charger le dataset
fmnist = fetch_openml(name="Fashion-MNIST", version=1)
# on peut commencer avec une partie du dataset
n_samples = 4000
data = fmnist.data[:n_samples]
target = fmnist.target[:n_samples]
# on répartit les données en training / test avec un ratio
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33)
```

We normalize the data by dividing each pixel value by 255. 

```python	
# normalization
x_train = x_train / 255
x_test = x_test / 255 
```
We reshape x_train and x_test to 2D array of 28x28 format without forgetting to convert to a numpy array to be allow to reshape. 

```python	
import numpy as np

# Have to convert to numpy to reshape
x_train = x_train.to_numpy().reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.to_numpy().reshape((x_test.shape[0], 28, 28, 1))
```
We also perform one-hot encoding on the labels using `to_categorical` from tensorflow.keras.utils.  

```python
import tensorflow.keras as keras

# one hot encoding => Transform to matrix of binary value
num_categories = 10
y_train = keras.utils.to_categorical(y_train) 
y_test = keras.utils.to_categorical(y_test)
```

## Building the Model
Our CNN model consists of a series of convolutional layers, pooling layers, and dense layers. We use `keras.Sequential` to create a sequential model. Our model consists of the following layers:

- `Conv2D` layer with 64 filters and a kernel size of (3,3). The activation function used is ReLU. The input shape is (28,28,1).
- `MaxPooling2D` layer with a pool size of (2,2).
- `Flatten` layer to flatten the output of the previous layer.
- `Dense` layer with 128 units and a ReLU activation function.
- `Dense` layer with 10 units (one for each category) and a softmax activation function.  

We compile the model using `compile` and specify the optimizer as adam, the loss function as `categorical_crossentropy`, and the metric as `accuracy`.

```python	
model = keras.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)), # Avoid overfitting
    keras.layers.Flatten(), # convert output of the previous layer into 1D vector
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Training and Evaluating the Model
We train the model using `fit` and specify the training data, the labels, the number of epochs, the batch size, and the validation split.

```python
# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.33)
```

## Evaluating the Model
In this part of the code, we are displaying the `history` of the model. The `history` variable contains a dictionary with four keys: `loss`, `accuracy`, `val_loss`, and `val_accuracy`. Each key corresponds to a list of the respective metric values recorded during the training of the model.

The first plot shows the accuracy of the model over epochs for both the training and validation sets. The `plt.plot()` function is used to plot the values of `accuracy` and `val_accuracy` over epochs. The `plt.title()`, `plt.ylabel()`, `plt.xlabel()`, and `plt.legend()` functions are used to set the title, x-axis label, y-axis label, and legend for the plot.

The second plot shows the loss of the model over epochs for both the training and validation sets. The `plt.plot()` function is used to plot the values of `loss` and `val_loss` over epochs. The `plt.title()`, `plt.ylabel()`, `plt.xlabel()`, and `plt.legend()` functions are used to set the title, x-axis label, y-axis label, and legend for the plot.

By analyzing these plots, we can determine if our model is overfitting, underfitting, or if it is appropriately fitting the training data. If the model is overfitting, the training accuracy will increase while the validation accuracy will decrease over time. If the model is underfitting, both training and validation accuracy will be low. If the model is appropriately fitting, training accuracy and validation accuracy will be high and close to each other. Additionally, we can also observe the loss of the model, where a decrease in loss signifies an improvement in the model's performance.

```python
# display history keys and plots
# display history for acc
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# display history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

![py01](https://i.imgur.com/NpvzSIt.png)

![py02](https://i.imgur.com/U5lrS4m.png)

## Predicting New Images
We can predict new images using our trained model by passing the image to the model using predict. We then use argmax to determine the category with the highest probability.
We set category_names to show the prediction category.

```python
# Define category names
category_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

predictions = model.predict(x_test)
```
```python
predicted_classes = predictions.argmax(axis=1)
```

## Displaying the Predictions
We can display the predictions of the model by using matplotlib. We use `np.random.randint` to generate a random integer between 0 and the number of images in the test set. We then use the random integer to select an image from the test set. We then reshape the image to 28x28 format and display it using `plt.imshow`. We then set the x and y ticks to be empty and set the label to be the predicted category. We then use `plt.show()` to display the image.

```python
import numpy as np
plt.figure(figsize=(10,10))
for i in range(10):
    idx = np.random.randint(len(x_test))
    image = x_test[idx].reshape((28, 28))
    plt.subplot(5,5,i+1)
    plt.imshow(image, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Predicted: {}'.format(predicted_classes[idx]))
    plt.show()
```

