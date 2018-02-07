import numpy as np
import matplotlib.pyplot as plt

# load data into Keras format
import keras

train_data = np.loadtxt('../data/training_data.txt', delimiter=' ', skiprows=1)
imp = np.loadtxt('../data/order.txt', delimiter=' ').astype(int)

y_train = train_data[:, 0]
x_train = train_data[:, 1:]

#h = 1000
#x_train = np.zeros((len(x_temp), h))
#count = 0
#while count < h:
#	for i in range(len(x_temp)):
#		x_train[i, count] = x_temp[i, count]
#	count += 1

# we'll need to one-hot encode the labels
y_train = keras.utils.np_utils.to_categorical(y_train)

# N O R M A L I Z E
x_train = np.divide(x_train, np.amax(x_train))


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras import regularizers

# sample model
# note: what is the difference between 'same' and 'valid' padding?
# Take a look at the outputs to understand the difference, or read the Keras documentation!
model = Sequential()
model.add(Dense(3, input_shape=(1000,)))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(20))
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
history = model.fit(x_train, y_train, epochs=6, batch_size=32,
                    validation_split=0.2)

print(model.count_params())

print(model.evaluate(x=x_train, y=y_train))