import time

from tensorflow._api.v1.keras.models import Sequential

from tensorflow._api.v1.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from tensorflow._api.v1.keras.callbacks import TensorBoard

# Saving the training data for future.It will not replace with but it will append

import pickle

from tensorflow.python.keras.optimizers import Adadelta

feature = pickle.load(open('feature.pickle', 'rb'))

label = pickle.load(open('label.pickle', 'rb'))

feature = feature / 255.0
import time

c = 0
dense_layers = [2]

layer_sizes = [128]

cnn_layers = [1]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for cnn_layer in cnn_layers:
            c += 1
            name = '{}_conv_{}_Nodes_{}_Dense_{}_{}'.format(cnn_layer, layer_size, dense_layer, int(time.time()), c)
            tensorboard = TensorBoard(log_dir='logs/{}'.format(name))
            print(name)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=feature.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(cnn_layer - 1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(feature, label, batch_size=50, validation_split=0.1, epochs=1, callbacks=[tensorboard])
            # Epoch that is how many times you will train your neural network if epoch>you will get better accuracy

            # tensorboard --logdir='logs/'
model.save('Final_1d_128ls_4cnn')
