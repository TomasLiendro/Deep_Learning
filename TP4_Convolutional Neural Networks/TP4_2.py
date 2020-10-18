import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.cifar10 as cifar10

import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import tensorflow.keras.regularizers as regression
import tensorflow.keras.optimizers as opt
import tensorflow.keras.losses as losses
import tensorflow.keras.activations as activations
import tensorflow.keras.initializers as initializers

import tensorflow as tf


def xor_acc(y_true=None, y_pred=None):
    acc = tf.reduce_mean(tf.cast(tf.less_equal(tf.abs(y_true - y_pred), 0.1), tf.float32))
    return acc


def TP2_3a(x_train, y_train, x_test, y_test):  # MSE
    n_train_data = x_train.shape[0]  # Cantidad de datos que se usarán para entrenar
    nclases = 10  # salida
    nintermedia = 100  # intermedia
    batch_size = 200  # Batch size
    n_epocas = 50  # Cantidad de épocas
    learning_rate = 1e-1  # Learning rate
    reg_lambda = 1e-3  # Coeficiente de regularización

    # acondiciono los datos de training
    im_shape = x_train.shape[1:]
    xtr = np.reshape(x_train[:n_train_data],
                     (x_train[:n_train_data].shape[0], np.prod(im_shape)))  # Los datos de train como vector

    x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
    std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

    xtr = (xtr - x_mean) / std
    ytr = y_train[:n_train_data]
    yy_tr = np.zeros((ytr.shape[0], nclases))  # esto es un vector de train_data x 10 para representar las clases
    yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # vector de train_data x 10

    # Acondiciono los datos de testing:
    n_train_data_t = int(n_train_data / 10)  # los datos que se van a tomar para el testing
    im_shape_test = x_test.shape[1:]
    xt = np.reshape(x_test[:n_train_data_t],
                    (x_test[:n_train_data_t].shape[0], np.prod(im_shape_test)))  # Los datos de test como vector
    xt = (xt - x_mean) / std
    yt = y_test[:n_train_data_t]

    yy_t = np.zeros((yt.shape[0], nclases))  # esto es un vector de train_data x 10 para representar las clases
    yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

    model = models.Sequential()

    # Primera capa:
    model.add(layers.Dense(units=nintermedia, input_shape=(xtr.shape[1],), activation=activations.sigmoid,
                           kernel_regularizer=reg.l2(reg_lambda), use_bias=True))
    # capa:
    model.add(layers.Dense(units=nclases, activation=activations.linear, kernel_regularizer=reg.l2(reg_lambda),
                           use_bias=True))
    model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer, loss="mse", metrics=["accuracy"])
    results = model.fit(xtr, yy_tr, batch_size=batch_size, epochs=n_epocas, verbose=2, validation_data=(xt, yy_t))

    plt.figure()
    plt.plot(np.arange(n_epocas), results.history['val_accuracy'], 'r*-')
    plt.plot(np.arange(n_epocas), results.history['accuracy'], 'k*-')
    plt.legend(['Datos de validación', 'Datos de entrenamiento'])
    plt.title('TP2 - Ej3: CIFAR-10, MSE')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    # plt.ylim([10, 50])
    plt.savefig('TP4_2aAcc.pdf')
    plt.figure()
    plt.plot(np.arange(n_epocas), results.history['val_loss'], 'r*-')
    plt.plot(np.arange(n_epocas), results.history['loss'], 'k*-')
    plt.title('TP2 - Ej3: CIFAR-10, MSE')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend(['Datos de validación', 'Datos de entrenamiento'])
    # plt.ylim([10, 50])
    plt.savefig('TP4_2aLoss.pdf')


def TP2_3b(x_train, y_train, x_test, y_test):  # SVM
    n_train_data = x_train.shape[0]  # Cantidad de datos que se usarán para entrenar
    nclases = 10  # salida
    nintermedia = 100  # intermedia
    batch_size = 200  # Batch size
    n_epocas = 50  # Cantidad de épocas
    learning_rate = 1e-3  # Learning rate
    reg_lambda = 1e-2  # Coeficiente de regularización

    # acondiciono los datos de training
    im_shape = x_train.shape[1:]
    xtr = np.reshape(x_train[:n_train_data],
                     (x_train[:n_train_data].shape[0], np.prod(im_shape)))  # Los datos de train como vector

    x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
    std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

    xtr = (xtr - x_mean) / std
    ytr = y_train[:n_train_data]
    yy_tr = np.zeros((ytr.shape[0], nclases))  # Esto es un vector de train_data x 10 para representar las clases
    yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # Vector de train_data x 10

    # Acondiciono los datos de testing:
    n_train_data_t = int(n_train_data / 10)  # los datos que se van a tomar para el testing
    im_shape_test = x_test.shape[1:]
    xt = np.reshape(x_test[:n_train_data_t],
                    (x_test[:n_train_data_t].shape[0], np.prod(im_shape_test)))  # Los datos de test como vector
    xt = (xt - x_mean) / std
    yt = y_test[:n_train_data_t]

    yy_t = np.zeros((yt.shape[0], nclases))  # esto es un vector de train_data x 10 para representar las clases
    yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

    model = models.Sequential()

    # Primera capa:
    model.add(layers.Dense(units=nintermedia, input_shape=(xtr.shape[1],), activation=activations.sigmoid,
                           kernel_regularizer=reg.l2(reg_lambda), use_bias=True))
    # capa:
    model.add(layers.Dense(units=nclases, activation=activations.linear, kernel_regularizer=reg.l2(reg_lambda),
                           use_bias=True))
    model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer, loss=losses.hinge, metrics=["accuracy"])
    results = model.fit(xtr, yy_tr, batch_size=batch_size, epochs=n_epocas, verbose=2, validation_data=(xt, yy_t))

    plt.figure()
    plt.plot(np.arange(n_epocas), results.history['val_accuracy'], 'r*-')
    plt.plot(np.arange(n_epocas), results.history['accuracy'], 'k*-')
    plt.legend(['Datos de validación', 'Datos de entrenamiento'])
    plt.title('TP2 - Ej3: CIFAR-10, SVM')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    # plt.ylim([10, 50])
    plt.savefig('TP4_2bAcc.pdf')
    plt.figure()
    plt.plot(np.arange(n_epocas), results.history['val_loss'], 'r*-')
    plt.plot(np.arange(n_epocas), results.history['loss'], 'k*-')
    plt.title('TP2 - Ej3: CIFAR-10, SVM')
    plt.legend(['Datos de validación', 'Datos de entrenamiento'])

    plt.xlabel('Época')
    plt.ylabel('Loss')
    # plt.ylim([10, 50])
    plt.savefig('TP4_2bLoss.pdf')


def TP2_3c(x_train, y_train, x_test, y_test):  # CCE
    n_train_data = x_train.shape[0]  # Cantidad de datos que se usarán para entrenar
    nclases = 10  # salida
    nintermedia = 100  # intermedia
    batch_size = 50  # Batch size
    n_epocas = 50  # Cantidad de épocas
    learning_rate = 1  # Learning rate
    reg_lambda = 1e-4  # Coeficiente de regularización

    # acondiciono los datos de training
    im_shape = x_train.shape[1:]
    xtr = np.reshape(x_train[:n_train_data],
                     (x_train[:n_train_data].shape[0], np.prod(im_shape)))  # Los datos de train como vector

    x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
    std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

    xtr = (xtr - x_mean) / std
    ytr = y_train[:n_train_data]
    yy_tr = np.zeros((ytr.shape[0], nclases))  # Esto es un vector de train_data x 10 para representar las clases
    yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # Vector de train_data x 10

    # Acondiciono los datos de testing:
    n_train_data_t = int(n_train_data / 10)  # los datos que se van a tomar para el testing
    im_shape_test = x_test.shape[1:]
    xt = np.reshape(x_test[:n_train_data_t],
                    (x_test[:n_train_data_t].shape[0], np.prod(im_shape_test)))  # Los datos de test como vector
    xt = (xt - x_mean) / std
    yt = y_test[:n_train_data_t]

    yy_t = np.zeros((yt.shape[0], nclases))  # esto es un vector de train_data x 10 para representar las clases
    yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

    model = models.Sequential()

    # Primera capa:
    model.add(layers.Dense(units=nintermedia, input_shape=(xtr.shape[1],), activation=activations.sigmoid,
                           kernel_regularizer=reg.l2(reg_lambda), use_bias=True,
                           kernel_initializer=initializers.RandomNormal(stddev=3)))
    # capa:
    model.add(layers.Dense(units=nclases, activation=activations.linear, kernel_regularizer=reg.l2(reg_lambda),
                           use_bias=True, kernel_initializer=initializers.RandomNormal(stddev=1)))
    model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer, loss=losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    results = model.fit(xtr, yy_tr, batch_size=batch_size, epochs=n_epocas, verbose=2, validation_data=(xt, yy_t))

    plt.figure()
    plt.plot(np.arange(n_epocas), results.history['val_accuracy'], 'r*-')
    plt.plot(np.arange(n_epocas), results.history['accuracy'], 'k*-')
    plt.legend(['Datos de validación', 'Datos de entrenamiento'])
    plt.title('TP2 - Ej3: CIFAR-10, CCE')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    # plt.ylim([10, 50])
    plt.savefig('TP4_2cAcc.pdf')
    plt.figure()
    plt.plot(np.arange(n_epocas), results.history['val_loss'], 'r*-')
    plt.plot(np.arange(n_epocas), results.history['loss'], 'k*-')
    plt.title('TP2 - Ej3: CIFAR-10, CCE')
    plt.legend(['Datos de validación', 'Datos de entrenamiento'])
    plt.xlabel('Época')
    plt.ylabel('Loss')
    # plt.ylim([10, 50])
    plt.savefig('TP4_2cLoss.pdf')


def TP2_3All(x_train, y_train, x_test, y_test):
    n_train_data = x_train.shape[0]  # Cantidad de datos que se usarán para entrenar
    nclases = 10  # salida
    nintermedia = 100  # intermedia
    batch_size = 50  # Batch size
    n_epocas = 50  # Cantidad de épocas
    learning_rate = 1  # Learning rate
    reg_lambda = 1e-4  # Coeficiente de regularización

    # acondiciono los datos de training
    im_shape = x_train.shape[1:]
    xtr = np.reshape(x_train[:n_train_data],
                     (x_train[:n_train_data].shape[0], np.prod(im_shape)))  # Los datos de train como vector

    x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
    std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

    xtr = (xtr - x_mean) / std
    ytr = y_train[:n_train_data]
    yy_tr = np.zeros((ytr.shape[0], nclases))  # Esto es un vector de train_data x 10 para representar las clases
    yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # Vector de train_data x 10

    # Acondiciono los datos de testing:
    n_train_data_t = int(n_train_data / 10)  # los datos que se van a tomar para el testing
    im_shape_test = x_test.shape[1:]
    xt = np.reshape(x_test[:n_train_data_t],
                    (x_test[:n_train_data_t].shape[0], np.prod(im_shape_test)))  # Los datos de test como vector
    xt = (xt - x_mean) / std
    yt = y_test[:n_train_data_t]

    yy_t = np.zeros((yt.shape[0], nclases))  # esto es un vector de train_data x 10 para representar las clases
    yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10
    # CCE
    model = models.Sequential()
    model.add(layers.Dense(units=nintermedia, input_shape=(xtr.shape[1],), activation=activations.sigmoid,
                           kernel_regularizer=reg.l2(reg_lambda), use_bias=True,
                           kernel_initializer=initializers.RandomNormal(stddev=3)))
    model.add(layers.Dense(units=nclases, activation=activations.linear, kernel_regularizer=reg.l2(reg_lambda),
                           use_bias=True, kernel_initializer=initializers.RandomNormal(stddev=1)))
    model.summary()
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer, loss=losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    results = model.fit(xtr, yy_tr, batch_size=batch_size, epochs=n_epocas, verbose=2, validation_data=(xt, yy_t))

    plt.figure(100)
    plt.plot(np.arange(n_epocas), results.history['val_accuracy'], '*-')
    plt.figure(101)
    plt.plot(np.arange(n_epocas), results.history['val_loss'], '*-')


    # SVM
    model = models.Sequential()
    learning_rate = 1e-3  # Learning rate
    model.add(layers.Dense(units=nintermedia, input_shape=(xtr.shape[1],), activation=activations.sigmoid,
                           kernel_regularizer=reg.l2(reg_lambda), use_bias=True))
    model.add(layers.Dense(units=nclases, activation=activations.linear, kernel_regularizer=reg.l2(reg_lambda),
                           use_bias=True))
    model.summary()
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer, loss=losses.hinge, metrics=["accuracy"])
    results = model.fit(xtr, yy_tr, batch_size=batch_size, epochs=n_epocas, verbose=2, validation_data=(xt, yy_t))
    plt.figure(100)
    plt.plot(np.arange(n_epocas), results.history['val_accuracy'], '*-')
    plt.figure(101)
    plt.plot(np.arange(n_epocas), results.history['val_loss'], '*-')


    # MSE
    model = models.Sequential()
    learning_rate = 1e-1  # Learning rate
    model.add(layers.Dense(units=nintermedia, input_shape=(xtr.shape[1],), activation=activations.sigmoid,
                           kernel_regularizer=reg.l2(reg_lambda), use_bias=True))
    model.add(layers.Dense(units=nclases, activation=activations.linear, kernel_regularizer=reg.l2(reg_lambda),
                           use_bias=True))
    model.summary()
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer, loss="mse", metrics=["accuracy"])
    results = model.fit(xtr, yy_tr, batch_size=batch_size, epochs=n_epocas, verbose=2, validation_data=(xt, yy_t))

    plt.figure(100)
    plt.plot(np.arange(n_epocas), results.history['val_accuracy'], '*-')
    plt.legend(['CCE', 'SVM', 'MSE'])
    plt.title('TP2 - Ej3: CIFAR-10')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    # plt.ylim([10, 50])
    plt.savefig('TP4_23AllAcc.pdf')
    plt.figure(101)
    plt.plot(np.arange(n_epocas), results.history['val_loss'], '*-')
    plt.title('TP2 - Ej3: CIFAR-10')
    plt.legend(['CCE', 'SVM', 'MSE'])
    plt.xlabel('Época')
    plt.ylabel('Loss')
    # plt.ylim([10, 50])
    plt.savefig('TP4_23AllLoss.pdf')


def TP2_4(x_train, y_train, x_test, y_test):
    n_train_data = x_train.shape[0]  # Cantidad de datos que se usarán para entrenar
    nclases = 10  # salida
    nintermedia = 100  # intermedia
    batch_size = 50  # Batch size
    n_epocas = 50  # Cantidad de épocas
    learning_rate = 1e-1  # Learning rate
    reg_lambda = 1e-4  # Coeficiente de regularización

    # acondiciono los datos de training
    im_shape = x_train.shape[1:]
    xtr = np.reshape(x_train[:n_train_data],
                     (x_train[:n_train_data].shape[0], np.prod(im_shape)))  # Los datos de train como vector

    x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
    std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

    xtr = (xtr - x_mean) / std
    ytr = y_train[:n_train_data]
    yy_tr = np.zeros((ytr.shape[0], nclases))  # Esto es un vector de train_data x 10 para representar las clases
    yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # Vector de train_data x 10

    # Acondiciono los datos de testing:
    n_train_data_t = int(n_train_data / 10)  # los datos que se van a tomar para el testing
    im_shape_test = x_test.shape[1:]
    xt = np.reshape(x_test[:n_train_data_t],
                    (x_test[:n_train_data_t].shape[0], np.prod(im_shape_test)))  # Los datos de test como vector
    xt = (xt - x_mean) / std
    yt = y_test[:n_train_data_t]

    yy_t = np.zeros((yt.shape[0], nclases))  # esto es un vector de train_data x 10 para representar las clases
    yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

    model = models.Sequential()

    # Primera capa:
    model.add(layers.Dense(units=nintermedia, input_shape=(xtr.shape[1],), activation=activations.sigmoid,
                           kernel_regularizer=reg.l2(reg_lambda), use_bias=True,
                           kernel_initializer=initializers.RandomNormal(stddev=3)))
    # Segunda capa:
    model.add(layers.Dense(units=nclases, activation=activations.linear, kernel_regularizer=reg.l2(reg_lambda),
                           use_bias=True, kernel_initializer=initializers.RandomNormal(stddev=1)))
    model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer, loss=losses.CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    results = model.fit(xtr, yy_tr, batch_size=batch_size, epochs=n_epocas, verbose=2, validation_data=(xt, yy_t))

    plt.figure()
    plt.plot(np.arange(n_epocas), results.history['val_accuracy'], 'r*-')
    plt.plot(np.arange(n_epocas), results.history['accuracy'], 'k*-')
    plt.legend(['Datos de validación', 'Datos de entrenamiento'])
    plt.title('TP2 - Ej4: CIFAR-10, Embeddings')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    # plt.ylim([10, 50])
    plt.savefig('TP4_24Acc.pdf')
    plt.figure()
    plt.plot(np.arange(n_epocas), results.history['val_loss'], 'r*-')
    plt.plot(np.arange(n_epocas), results.history['loss'], 'k*-')
    plt.title('TP2 - Ej4: CIFAR-10, Embeddings')
    plt.legend(['Datos de validación', 'Datos de entrenamiento'])
    plt.xlabel('Época')
    plt.ylabel('Loss')
    # plt.ylim([10, 50])
    plt.savefig('TP4_24Loss.pdf')


def TP2_6a(x_train, y_train, x_test=None, y_test=None):
    n_train_data = x_train.shape[0]  # Cantidad de datos que se usarán para entrenar
    nclases = 1  # salida
    nintermedia = 2  # intermedia
    batch_size = n_train_data  # Batch size
    n_epocas = 500  # Cantidad de épocas
    learning_rate = 1e-1  # Learning rate
    reg_lambda = 0  # Coeficiente de regularización

    # acondiciono los datos de training
    im_shape = x_train.shape[1:]
    xtr = np.reshape(x_train[:n_train_data],
                     (x_train[:n_train_data].shape[0], np.prod(im_shape)))  # Los datos de train como vector

    ytr = y_train[:n_train_data]

    # Acondiciono los datos de testing:
    model = models.Sequential()

    # Primera capa:
    x = layers.Input(shape=(2,))
    l1 = model.add(layers.Dense(units=nintermedia, input_shape=(xtr.shape[1],), activation=activations.tanh,
                                kernel_regularizer=reg.l2(reg_lambda), use_bias=True))
    # capa:
    l2 = model.add(layers.Dense(units=nclases, activation=activations.tanh, use_bias=True))
    # modelo = tf.keras.Model(inputs=x, outputs=l2)
    # model.summary()

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer, loss=losses.MSE, metrics=[xor_acc, 'acc', 'mse'])
    results = model.fit(xtr, ytr, batch_size=batch_size, epochs=n_epocas, verbose=2, validation_data=(xtr, ytr))

    plt.figure()
    # plt.plot(np.arange(n_epocas), results.history['val_accuracy'], 'r*-')
    plt.plot(np.arange(n_epocas), results.history['xor_acc'], '*-')
    # plt.plot(np.arange(n_epocas), results.history['acc'], '*-')

    plt.title('TP2 - Ej6: XOR')
    plt.legend(['Datos de entrenamiento'])
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.savefig('TP4_26Acc.pdf')
    # plt.plot(np.arange(n_epocas), results.history['mse'], '*-')
    plt.figure()
    # plt.plot(np.arange(n_epocas), results.history['val_loss'], 'r*-')
    plt.plot(np.arange(n_epocas), results.history['loss'], 'k*-')
    plt.title('TP2 - Ej6: XOR')
    plt.legend(['Datos de entrenamiento'])
    plt.xlabel('Época')
    plt.ylabel('Loss')
    # plt.ylim([10, 50])
    plt.savefig('TP4_26Loss.pdf')


(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # Cargo los datos de CIFAR-10
TP2_3a(x_train, y_train, x_test, y_test)
TP2_3b(x_train, y_train, x_test, y_test)
TP2_3c(x_train, y_train, x_test, y_test)  #
# TP2_3All(x_train, y_train, x_test, y_test)  # Todos

# TP2_4(x_train, y_train, x_test, y_test)  #
#
# x_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
# y_train = np.array([[1], [-1], [-1], [1]])

# TP2_6a(x_train, y_train)
plt.show()
