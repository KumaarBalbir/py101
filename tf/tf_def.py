import pandas as pd
import numpy as np
from tensorflow import constant, add
from tensorflow import constant
from tensorflow import ones, reduce_sum
from tensorflow import ones, matmul, multiply
import tensorflow as tf

# 0D tensor
d0 = tf.ones((1,))

# 1D tensor
d1 = tf.ones((2,))  # TODO: How it is different from d0?

# 2D tensor
d2 = tf.ones((2, 2))

# 3D tensor
d3 = tf.ones((2, 2, 2))

# convert tensor to numpy
print(d3.numpy())

# Defining constants in tf

# define a 2x3 constant tensor
a = constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = constant(3, shape=[2, 3])  # all values are 3

# Define a variable
a0 = tf.Variable([1, 2, 3, 4, 5, 6], dtype=tf.float32)
a1 = tf.Variable([1, 2, 3, 4, 5, 6], dtype=tf.int32)

# define a constant
b = tf.constant(2, tf.float32)

# Compute their product
c0 = tf.multiply(a0, b)
c1 = a0*b

# apply addition operator

# Define 0-dim tensors
A0 = constant([1])
B0 = constant([2])

# Define a 1-dim tensor
A1 = constant([1, 2])
B1 = constant([3, 4])

# Define a 2-dim tensor
A2 = constant([[1, 2], [3, 4]])
B2 = constant([[5, 6], [7, 8]])

# perform tensor additon with add()
C0 = add(A0, B0)
C1 = add(A1, B1)
C2 = add(A2, B2)

# multiply(): element wise multiplication, tensors being multiplied must have the same shape e.g. [1,2,3] and [3,4,5] or [1,2] and [3,4]
# matmul(): matrix multiplication, e.g. matmul(A,B) col of A should be equal to row of B


A0 = ones(1)
A31 = ones([3, 1])  # fill with 1 in 3 rows and 1 column
A34 = ones([3, 4])
A43 = ones([4, 3])

# valid operations: multiply(A0,A0), multiply(A31,A31), multiply(A34,A34), multiply(A43,A43), matmul(A34,A43)
# invalid operation: matmul(A43, A43)

# Reduce sum
A = ones([2, 3, 4])
B = reduce_sum(A)  # B = 24, sums over all dimension

# sum over a specific dimension
B0 = reduce_sum(A, 0)  # B0 = [6,6,6,6]
B1 = reduce_sum(A, 1)  # B1 = [6,6,6,6]
B2 = reduce_sum(A, 2)  # B2 = [6,6,6,6]

# TODO: Check above values

# Advanced Operations: gradient(), reshape(), random()
# gradient(): slope of a function at a point
# reshape(): change the shape of a tensor (e.g. 10x10 to 100x1)
# random(): populates tensors with entries drawn from a probability distribution

# **************** Gradients in tensorflow ******************

# define x
x = tf.Variable(-1.0)

# define y within insance of GradientTape
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.multiply(x, x)

# evaluate the grad of y at x=-1
g = tape.gradient(y, x)
print(g.numpy())  # -2

# **************** Reshaping tensors ******************

# generate grayscale image
gray = tf.random.uniform([2, 2], maxval=255, dtype=tf.int32)

# reshape grayscale image
gray = tf.reshape(gray, [2*2, 1])  # [2,2] -> [4,1]

# color image
color = tf.random.uniform([2, 2, 3], maxval=255, dtype=tf.int32)

# reshape color image
color = tf.reshape(color, [2*2, 3])  # [2,2,3] -> [4,3]

# Import and convert data

# other options: read_json, read_html, read_excel
housing = pd.read_csv('kc_housing_data.csv', sep=',')

# convert to numpy array
housing = np.array(housing)

# convert price column to float32
price = np.array(housing['price'], dtype=np.float32)

# convert price column to float32
price = tf.cast(housing['price'], tf.float32)

# ************************ LOSS FUNCTIONS IN TENSORFLOW **********************
# loss functions accessible from: tf.keras.losses()
# MSE: tf.keras.losses.mse()
# MAE: tf.keras.losses.mae()
# Huber: tf.keras.losses.Huber() # TODO: What is the use of huber error?

# Define a loss function

loss = tf.keras.losses.mse(targets, predictions)

# define a linear regression model


def linear_regression(intercept, slope=slope, features=features):
    return intercept + features*slope

# define a loss function to compute the MSE


def loss_function(intercept, slope, targets=targets, features=features):
    predictions = linear_regression(intercept, slope, features)

    # return the loss
    return tf.keras.losses.mse(targets, predictions)


# linear reg in tensorflow
price = np.array(housing['price'], dtype=np.float32)
size = np.array(housing['sqft_living'], dtype=np.float32)

intercept = tf.Variable(0.1, np.float32)
slope = tf.Variable(0.1, np.float32)

# define an optimization operation
opt = tf.keras.optimizers.Adam()

# Minimize the loss function and print the loss
for j in range(1000):
    opt.minimize(lambda: loss_function(intercept=intercept,
                 slope=slope), var_list=[intercept, slope])
    print(loss_function(intercept=intercept, slope=slope))

# print trained parameters
print(intercept.numpy(), slope.numpy())


# ************************** BATCH TRAINING IN TENSORFLOW ********************
for batch in pd.read_csv('kc_housing_data.csv', sep=',', chunksize=100):
    price_batch = np.array(batch['price'], np.float32)
    size_batch = np.array(batch['sqft_living'], np.float32)

    # minimize loss function
    opt.minimize(lambda: loss_function(intercept=intercept, slope=slope,
                 targets=price_batch, features=size_batch), var_list=[intercept, slope])


# print trained parameters
print(intercept.numpy(), slope.numpy())

# ************************************* Low Level: DENSE Layer in tensorflow ********************
# define inputs (features)
inputs = tf.constant([[1, 35]])  # dim: [1,2]

# define weights
weights = tf.Variable([[0.5], [0.5]])  # dim: [2,1]

# define bias
bias = tf.Variable([0.5])

# low level dense layer
# multiply inputs with weights(features)
product = tf.matmul(inputs, weights)  # dim: [1,1]

# add bias
output = product + bias  # dim: [1,1]

# take activation
dense = tf.keras.activations.sigmoid(output)


# *************************************** High Level: DENSE Layer in tensorflow ********************

# define inputs (features)
input_data = tf.constant([[1, 35]])  # dim: [1,2]
inputs = tf.constant(input_data, tf.float32)

# define first dense layer
dense1 = tf.keras.layers.Dense(10, activation='sigmoid')(inputs)

# define second dense layer
dense2 = tf.keras.layers.Dense(5, activation='sigmoid')(dense1)

# define output layer (pred layer)
outputs = tf.keras.layers.Desne(1, activation='sigmoid')(dense2)

# Activations function
# Low level: tf.keras.activations.<activation function>()
# Example: tf.keras.activations.sigmoid(x), tf.keras.activations.relu(x), tf.keras.activations.tanh(x), tf.keras.activations.softmax(x),etc

# High level: sigmoid(x), relu(x), tanh(x), softmax(x), etc

# ********************* Optimizers ****************************

# SGD: tf.keras.optimizers.SGD()
# Adam: tf.keras.optimizers.Adam()
# RMSprop: tf.keras.optimizers.RMSprop()
# Adagrad: tf.keras.optimizers.Adagrad()
# Adadelta: tf.keras.optimizers.Adadelta()
# Adamax: tf.keras.optimizers.Adamax()
# Nadam: tf.keras.optimizers.Nadam()

# Complete example


def model(bias, weights, features):
    product = tf.matmul(features, weights)
    return tf.keras.activations.sigmoid(product + bias)


def predict(bias, weights, features):
    return model(bias, weights, features)


def loss_function(targets, predictions):
    return tf.keras.losses.mse(targets, predictions)


pred = predict(bias=bias, weights=weights, features=input_data)
loss = loss_function(targets=price_batch, predictions=pred)

# Minimize the loss function with RMS prop
opt = tf.keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.9)
opt.minimize(lambda: loss, var_list=[bias, weights])


# ******************************** INITIALIZERS ****************************

# define 500x 500 random normal variable
weights = tf.Variable(tf.random.normal([500,500]))

# truncated random var 
weights = tf.Variable(tf.random.truncated_normal([500,500]))

# define dense layer with default initializers 
dense = tf.keras.layers.Dense(32,activation='relu') 

# define dense layer with zeroes initializer 
dense = tf.keras.layers.Dense(32, activation='relu',keral_initializer='zeroes') 

# ******************************* DROPOUT in NN *******************************
import numpy as np 
import tensorflow as tf 

# input data 
input_features = np.array([[2,3,4],[1,5,2]])
inputs = np.array(input_features, np.float32) 

# dense layer 1 
dense1 = tf.keras.layers.Dense(32,activation='relu')(inputs) 

# dense layer 2 
dense2 = tf.keras.layers.Dense(16, activation='relu')(dense1) 

# apply dropout operation
dropout1 = tf.keras.layers.Dropout(0.25)(dense2)

# output layer 
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dropout1)

# ************************************ NN with Keras : Sequential API **************
from tensorflow import keras 

# define a seq model 
model = keras.Sequential() 

# first hidden layer 
model.add(keras.layers.Dense(16, activation='relu', input_shape=(28*28,))) # TODO: how to provide shape?

# second hidden layer
model.add(keras.layers.Dense(8, activation='relu'))

# output layer 
model.add(keras.layers.Dense(4, activation='softmax'))

# compile the model
model.compile('adam', loss='categorical_crossentropy')

# summary of the model 
print(model.summary())

# ********************************** FUNCTIONAL API ************************
import tensorflow as tf 

# define model 1 input layer shape 
model1_inputs = tf.keras.Input(shape=(28*28,)) # TODO: what does (28*28,) mean? 

# define model 2 input layer shape 
model2_inputs = tf.keras.Input(shape=(10,)) 

# define layer 1 for model 1 
model1_layer1 = tf.keras.layers.Dense(12, activation='relu')(model1_inputs) 

# define layer 2 for model 1 
model1_layer2 = tf.keras.layers.Dense(4, activation='softmax')(model1_layer1)

# define layer 1 for model 2 
model2_layer1 = tf.keras.layers.Dense(8, activation='relu')(model2_inputs)

# define layer 2 for model 2 
model2_layer2 = tf.keras.layers.Dense(4, activation='softmax')(model2_layer1) 

# merge model 1 and model 2 
merged = tf.keras.layers.add([model1_layer2, model2_layer2])

# define a functional model 
model = tf.keras.Model(inputs = [model1_inputs,model2_inputs], outputs= merged)

# compile the model 
model.compile('adam', loss='categorical_crossentropy') 

# ******************************** Train a model **********************
# Define the model -> compile the model -> train
model.fit(features, labels, batch_size, epochs, validation_split)

# changing the metric 
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy']) 

# train using model.fit(..)

# evaluate the model 
model.evaluate(test)



