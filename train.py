''' Train input data for
    Sign Language Recognition Model
'''
#%%
#_____________________________________Import Part 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from prepocessing import preprocessing_dataset
from matplotlib import pyplot as plt
import pandas as pd

#%%
#________________________Data Pre-processing Part
    # Load Dataset from csv file 
dataset_train = pd.read_csv('input/sign_mnist_train.csv')
dataset_test = pd.read_csv('input/sign_mnist_test.csv')

#%%
    # Prepare for train data
images_4train, labels_4train = preprocessing_dataset(dataset_train)

    # Splitting the data into train(70%) and test(30%)
x_train, x_test, y_train, y_test = train_test_split(images_4train, labels_4train, test_size = 0.3, random_state = 42)
x_train = x_train / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#%%
#__________________________________CNN Model Part 
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Neural Network config
batch_size = 128
num_classes = 24
epochs = 100 

# CNN Model 
model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))
model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, 
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

#%%
#___________________________________Training Part
history = model.fit(x_train, y_train, 
                    validation_data = (x_test, y_test), 
                    epochs=epochs, 
                    batch_size=batch_size)

#%%
#_____________________________Evaluate Model Part
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.show()

#%%
images_4test, labels_4test = preprocessing_dataset(dataset_test)
images_4test = images_4test.reshape(images_4test.shape[0], 28, 28, 1)
y_pred = model.predict(images_4test)
accuracy_score(labels_4test, y_pred.round())

#%%
# Save the model for use 
# Save the model with h5py
folder = 'slr_models/'
model_name = 'slr_model_v3'
model_json = model.to_json()
with open(folder + model_name + '.json','w') as file:
    file.write(model_json)
model.save_weights(folder + model_name + '.h5')
print("Saved model to disk")
