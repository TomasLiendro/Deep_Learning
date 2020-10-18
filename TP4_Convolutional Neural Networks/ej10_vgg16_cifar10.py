import tensorflow as tf
from tensorflow.keras.datasets import cifar10,cifar100
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(x_train,y_train),(x_test,y_test)=cifar10.load_data()
#Preprocesamiento
img_rows, img_cols , channels= 32,32,3

#Augmentation
datagen=ImageDataGenerator(
    rotation_range=90,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)
#reshape into images
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
input_shape = (img_rows, img_cols, 1)
#Datos x
x_train=x_train.astype("float32")/255
x_train=np.reshape(x_train,(x_train.shape[0], np.prod(x_train.shape[1:])))
media_x=np.mean(x_train,axis=0)
x_train=(x_train-media_x)
#x_train1=x_train[:5000,:]

x_test=x_test.astype("float32")/255
x_test=np.reshape(x_test,(x_test.shape[0], np.prod(x_test.shape[1:])))
x_test=(x_test-media_x)
#x_test1=x_test[:500,:]

#Datos y
y_train=y_train.ravel()
yy_train=np.zeros((x_train.shape[0],10))
y=np.arange(x_train.shape[0])
yy_train[np.arange(x_train.shape[0]),y_train]=1
#yy_train1=yy_train[:5000,:]

y_test=y_test.ravel()
yy_test=np.zeros((x_test.shape[0],10))
ytest=np.arange(x_test.shape[0])
yy_test[np.arange(y_test.shape[0]),y_test]=1
#yy_test1=yy_test[:500,:]

x_trainC = x_train.reshape(-1, 32, 32, 3)
x_testC = x_test.reshape(-1, 32, 32, 3)
weight_decay=0.0005
model=tf.keras.models.Sequential()
#VGG16 - CIFAR10
model.add(tf.keras.layers.Conv2D(64,kernel_size=(3,3),padding="same",activation="relu",input_shape=x_trainC.shape[1:],kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(64,kernel_size=(3,3),padding="same",activation="relu",kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(128,kernel_size=(3,3),padding="same",activation="relu",kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Conv2D(128,kernel_size=(3,3),padding="same",activation="relu",kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(256,kernel_size=(3,3),padding="same",activation="relu",kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Conv2D(256,kernel_size=(3,3),padding="same",activation="relu",kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Conv2D(256,kernel_size=(3,3),padding="same",activation="relu",kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding="same",activation="relu",kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding="same",activation="relu",kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding="same",activation="relu",kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding="same",activation="relu",kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding="same",activation="relu",kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding="same",activation="relu",kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512,activation="relu",kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(yy_train.shape[1],activation="softmax"))
model.summary()

optimizer=tf.keras.optimizers.Adadelta(lr=1)
model.compile(optimizer,loss=tf.keras.losses.categorical_crossentropy,metrics=["accuracy"])

epocas=20
#entreno
results_conv=model.fit(x_trainC,yy_train,epochs=epocas,batch_size=64,verbose=2,validation_data=(x_testC,yy_test))


plt.figure()
plt.plot(np.arange(epocas),results_conv.history["accuracy"],label="Train")
plt.plot(np.arange(epocas),results_conv.history["val_accuracy"], label="Test")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.title("VGG16, CIFAR10")
plt.legend(loc='lower right', shadow=False)
plt.figure()
plt.plot(np.arange(epocas),results_conv.history["loss"], label="Train")
plt.plot(np.arange(epocas),results_conv.history["val_loss"], label="Test")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.title("VGG16, CIFAR10")
plt.legend(loc='lower right', shadow=False)
plt.show()

