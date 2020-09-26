import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.cifar10 as cifar10


def SVM(y_pred,y_real):  # Función de costo SVM
    y_real = np.argmax(y_real, axis=1)
    y_pred=y_pred.T
    delta=1
    real = y_pred[y_real, np.arange(y_real.shape[0])]
    margins = y_pred - real +delta
    margins[margins < 0] = 0
    margins[y_real, np.arange(y_real.shape[0])] = 0
    L = np.sum(margins, axis=0)
    return np.mean(L)


def gradSVM(y_pred,y_real):  # Derivada de la función de costo SVM
    delta=1
    y_real = np.argmax(y_real, axis=1)
    y_pred = y_pred.T
    real = y_pred[y_real, np.arange(y_real.shape[0])]
    margins = y_pred - real + delta
    margins[margins < 0] = 0
    margins[y_real, np.arange(y_real.shape[0])] = 0
    margins[margins > 0] = 1
    margins[y_real, np.arange(y_real.shape[0])] = -np.sum(margins, axis=0)
    return margins.T


def CCE(y_pred,y_real):  # Función de costo Softmax (CCE)
    y_real = np.argmax(y_real, axis=1)
    y = np.max(y_pred.T, axis=0)
    y_pred=y_pred.T-y
    y_pred=y_pred.T
    real=y_pred[np.arange(y_real.shape[0]),y_real]
    L=-real+np.log(np.sum(np.exp(y_pred),axis=1))
    loss=np.mean(L)
    return loss


def gradCCE(y_pred,y_real):  # Derivada de softmax (CCE)
    y_real = np.argmax(y_real, axis=1)
    y = np.max(y_pred.T, axis=0)
    y_pred = y_pred.T - y
    y_pred = y_pred.T
    inter=np.sum(np.exp(y_pred),axis=1)
    margins=(np.exp(y_pred)).T/(inter)
    margins=margins.T
    margins[np.arange(y_pred.shape[0]),y_real]-=1
    return margins


def MSE(S2, yy_real):  # Función de Costo MSE
    y = np.mean(np.sum((S2 - yy_real) ** 2, axis=1))
    return y


def gradMSE(S2, yy_real):  # Gradiente de la función de Costo MSE
    y = 2 * (S2-yy_real)
    return y


def sigmoide(X):  # Función de activación Sigmoide
    exp = np.exp(-X)
    exp = 1 + exp
    y = 1 / exp
    return y


def grad_sigmoide(x):  # Gradiente de la función de activación Sigmoide
    y = sigmoide(x) * (1 - sigmoide(x))
    return y


def accuracy(y_est, y_real):  # Cálculo de la precisión
    acc = np.zeros((y_est.shape[0], 1))
    Y_pred = np.argmax(y_est, axis=1)
    acc[Y_pred[:, np.newaxis] == y_real] = 1
    acc_sum = np.sum(acc, axis=0)
    y = acc_sum/y_est.shape[0]
    return y


nclases = 10            # salida
nintermedia = 100       # intermedia
batch_size = 256        # Batch size
n_epocas = 50           # Cantidad de épocas
learning_rate = 1e-6    # Learning rate
reg_lambda = 1e-3       # Coeficiente de regularización
n_train_data = 5000     # Cantidad de datos que se usarán para entrenar

(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # Cargo los datos de CIFAR-10

# acondiciono los datos de training
im_shape = x_train.shape[1:]
xtr = np.reshape(x_train[:n_train_data], (x_train[:n_train_data].shape[0], np.prod(im_shape)))   # Los datos de train como vector

x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
std = np.std(xtr, axis=0)[np.newaxis, :]    # STD de los datos de training

xtr = (xtr - x_mean) / std

xtr = np.hstack([np.ones((x_train[:n_train_data].shape[0], 1)), xtr])   # agrego una fila de 1's para el bias
ytr = y_train[:n_train_data]

yy_tr = np.zeros((ytr.shape[0], nclases))  # esto es un vector de train_data x 10 para representar las clases
yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # vector de train_data x 10

nro_batchs = int(ytr.shape[0] / batch_size)  # Cantidad de batches

# Acondiciono los datos de testing:
n_train_data_t = int(n_train_data/10)  # los datos que se van a tomar para el testing
im_shape_test = x_test.shape[1:]
xt = np.reshape(x_test[:n_train_data_t], (x_test[:n_train_data_t].shape[0], np.prod(im_shape_test)))  # Los datos de test como vector
xt = (xt - x_mean) / std
xt = np.hstack([np.ones((x_test[:n_train_data_t].shape[0], 1)), xt])  # agrego una fila de 1's para el bias
yt = y_test[:n_train_data_t]

yy_t = np.zeros((yt.shape[0], nclases))  # esto es un vector de train_data x 10 para representar las clases
yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10


# '''
# inicializo las W de cada capa
W1 = np.random.normal(0, 1, (xtr.shape[1], nintermedia)) * 0.01  # 3073 x 100, meto el primer bias
W2 = np.random.normal(0, 1, (W1.shape[1] + 1, nclases)) * 0.1  # 101 x 10 : meto el segundo bias

# Se inicializan los vectores para guardar información para plotear
train_acc = np.zeros((n_epocas, 1))
train_loss = np.zeros((n_epocas, 1))
test_acc = np.zeros((n_epocas, 1))
test_loss = np.zeros((n_epocas, 1))
epoca = []

# Comienza el entrenamiento:
for i in range(n_epocas):
    indice = xtr.shape[0]
    indice = np.arange(indice)
    np.random.shuffle(indice)
    loss = 0
    for j in range(nro_batchs):
        # Forward
        x_batch = xtr[indice[(j * batch_size):((j + 1) * batch_size)], :]  # de batch_size x 3073
        y_batch = ytr[indice[(j * batch_size):((j + 1) * batch_size)]]  # de batch_size x 1

        yyy_tr = yy_tr[indice[(j * batch_size):((j + 1) * batch_size)], :]  # de batch_size x 10

        # Capa 1:
        Y1 = np.dot(x_batch, W1)  # esto es un vector de batch_size x 101
        S1 = sigmoide(Y1)
        S1_prima = np.hstack([np.ones((x_batch.shape[0], 1)), S1])  # esto es un vector de batch_size x 101

        # Capa 2
        S2 = np.dot(S1_prima, W2)  # es un vector de batch_size x 10
        y_test_estimado = np.dot(xt, W1)
        y_test_estimado_S1 = sigmoide(y_test_estimado)
        y_test_estimado_S1_prima = np.hstack([np.ones((xt.shape[0], 1)), y_test_estimado_S1])
        y_test_estimado_S2 = np.dot(y_test_estimado_S1_prima, W2)

        # Calculo la regularización
        reg1 = np.sum(W1 * W1)
        reg2 = np.sum(W2 * W2)
        reg = reg1 + reg2

        # Backward
        # Capa 1
        grad = gradMSE(S2, yyy_tr)  # gradiente de la Función costo
        grad = grad + 2 * (np.sum(W1) + np.sum(W2)) * reg_lambda / 2

        grad_W2 = np.dot(S1_prima.T, grad) # delta W2
        grad = np.dot(grad, W2.T)
        grad = grad[:, 1:]

        # Capa 2
        grad = grad_sigmoide(Y1) * grad # gradiente de la Función de activación
        grad_W1 = np.dot(x_batch.T, grad)

        # Actualización de los pesos
        W1 = W1 - learning_rate * (grad_W1 + reg * W1)
        W2 = W2 - learning_rate * (grad_W2 + reg * W2)

        # Cálculo de la loss y el accuracy
        train_loss[i] += MSE(S2, yyy_tr) + (reg1 + reg2) * 0.5
        train_acc[i] += accuracy(S2, y_batch)
        test_loss[i] += MSE(y_test_estimado_S2, yy_t) + (reg1 + reg2) * 0.5
        test_acc[i] += accuracy(y_test_estimado_S2, yt)

    epoca.append(i)

    train_acc[i] = train_acc[i]/nro_batchs
    train_loss[i] = train_loss[i]/nro_batchs
    test_acc[i] = test_acc[i] / nro_batchs
    test_loss[i] = test_loss[i]/nro_batchs

    print('Epoca: ' + str(i) + '  Accuracy: ' + str(train_acc[i]) + '     Loss: ' + str(test_loss[i]))

plt.figure(1)
#
# plt.plot(epoca, train_acc,'r*')
# plt.plot(epoca, test_acc,'k*')
plt.plot(epoca, test_acc,'*')
# plt.legend(['Training data', 'Testing data'])
# plt.title('Evolución del accuracy - CIFAR10')
# plt.xlabel('Época')
# plt.ylabel('Accuracy')
# plt.savefig('Accuracy{}bz{}td{}.pdf'.format(n_epocas, batch_size, n_train_data))
# fig, ax1 = plt.subplots()
# plt.title('Evolución de la Loss - CIFAR10')
# ax1.set_xlabel('Época')
# ax1.set_ylabel('Loss', color='tab:red')
# # plt.legend(['Training data', 'Testing data'])
# ax1.tick_params(axis='y', labelcolor='r')
# ax1.plot(epoca, train_loss, color='tab:red', marker='*')
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'k'
# ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
# ax2.plot(epoca, test_loss, color=color, marker='*')
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('Loss_ep{}bz{}td{}.pdf'.format(n_epocas, batch_size, n_train_data))
# plt.show(block=0)
# plt.pause(1)
# '''


###############   Ahora cambio por la función de costo SVM  ##############################
# '''
nclases = 10            # salida
nintermedia = 100       # intermedia
batch_size = 128        # Batch size
n_epocas = 50           # Cantidad de épocas
learning_rate = 1e-5    # Learning rate
reg_lambda = 1e-3       # Coeficiente de regularización
n_train_data = 5000     # Cantidad de datos que se usarán para entrenar

nro_batchs = int(ytr.shape[0] / batch_size) # Cantidad de batches

# inicializo las W de cada capa
W1 = np.random.normal(0, 1, (xtr.shape[1], nintermedia)) * 0.01  # 3073 x 100, meto el primer bias
W2 = np.random.normal(0, 1, (W1.shape[1] + 1, nclases)) * 0.001 # 101 x 10 : meto el segundo bias

# Se inicializan los vectores para guardar información para plotear
train_acc = np.zeros((n_epocas, 1))
train_loss = np.zeros((n_epocas, 1))
test_acc = np.zeros((n_epocas, 1))
test_loss = np.zeros((n_epocas, 1))
epoca = []

# Comienza el entrenamiento:
for i in range(n_epocas):
    indice = xtr.shape[0]
    indice = np.arange(indice)
    np.random.shuffle(indice)
    loss = 0
    for j in range(nro_batchs):
        # Forward
        x_batch = xtr[indice[(j * batch_size):((j + 1) * batch_size)], :]  # de batch_size x 3073
        y_batch = ytr[indice[(j * batch_size):((j + 1) * batch_size)]]  # de batch_size x 1

        yyy_tr = yy_tr[indice[(j * batch_size):((j + 1) * batch_size)], :]  # de batch_size x 10

        # Capa 1:
        Y1 = np.dot(x_batch, W1)  # esto es un vector de batch_size x 101
        S1 = sigmoide(Y1)
        S1_prima = np.hstack([np.ones((x_batch.shape[0], 1)), S1])  # esto es un vector de batch_size x 101

        # Capa 2
        S2 = np.dot(S1_prima, W2)  # es un vector de batch_size x 10
        y_test_estimado = np.dot(xt, W1)
        y_test_estimado_S1 = sigmoide(y_test_estimado)
        y_test_estimado_S1_prima = np.hstack([np.ones((xt.shape[0], 1)), y_test_estimado_S1])
        y_test_estimado_S2 = np.dot(y_test_estimado_S1_prima, W2)

        # Calculo la regularización
        reg1 = np.sum(W1 * W1)
        reg2 = np.sum(W2 * W2)
        reg = reg1 + reg2

        # Backward
        # Capa 1
        grad = gradSVM(S2, yyy_tr)  # gradiente de la Función costo
        grad = grad + 2 * (np.sum(W1) + np.sum(W2)) * reg_lambda / 2

        grad_W2 = np.dot(S1_prima.T, grad) # delta W2
        grad = np.dot(grad, W2.T)
        grad = grad[:, 1:]

        # Capa 2
        grad = grad_sigmoide(Y1) * grad # gradiente de la Función de activación
        grad_W1 = np.dot(x_batch.T, grad)

        # Actualización de los pesos
        W1 = W1 - learning_rate * (grad_W1 + reg * W1)
        W2 = W2 - learning_rate * (grad_W2 + reg * W2)

        # Cálculo de la loss y el accuracy
        train_loss[i] += SVM(S2, yyy_tr) + (reg1 + reg2) * 0.5
        train_acc[i] += accuracy(S2, y_batch)
        test_loss[i] += SVM(y_test_estimado_S2, yt) + (reg1 + reg2) * 0.5
        test_acc[i] += accuracy(y_test_estimado_S2, yt)

    epoca.append(i)

    train_acc[i] = train_acc[i]/nro_batchs
    train_loss[i] = train_loss[i]/nro_batchs
    test_acc[i] = test_acc[i] / nro_batchs
    test_loss[i] = test_loss[i]/nro_batchs

    print('Epoca: ' + str(i) + '  Accuracy: ' + str(train_acc[i]) + '     Loss: ' + str(test_loss[i]))

# plt.figure(1)
# plt.plot(epoca, train_acc,'r*')
# plt.plot(epoca, test_acc,'k*')
plt.plot(epoca, test_acc,'*')
# plt.legend(['Training data', 'Testing data'])
# plt.title('Evolución del accuracy - CIFAR10')
# plt.xlabel('Época')
# plt.ylabel('Accuracy')
# plt.savefig('Accuracy{}bz{}td{}.pdf'.format(n_epocas, batch_size, n_train_data))
# fig, ax1 = plt.subplots()
#
# # plt.plot(epoca, train_loss, 'r-*')
# # plt.plot(epoca, test_loss, 'k-*')
# plt.title('Evolución de la Loss - CIFAR10')
# ax1.set_xlabel('Época')
# ax1.set_ylabel('Loss', color='tab:red')
# ax1.tick_params(axis='y', labelcolor='r')
# # plt.legend(['Training data', 'Testing data'])
#
# ax1.plot(epoca, train_loss, color='tab:red',marker='*')
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'k'
# ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
# ax2.plot(epoca, test_loss, color=color, marker='*')
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('Loss_ep{}bz{}td{}.pdf'.format(n_epocas, batch_size, n_train_data))
# # plt.ylim(0,train_loss[1])
# plt.show(block=0)
# plt.pause(1)
# '''


###############   Ahora cambio por la función de costo CCE  ##############################

nclases = 10            # salida
nintermedia = 100       # intermedia
batch_size = 128        # Batch size
n_epocas = 50           # Cantidad de épocas
learning_rate = 1e-5    # Learning rate
reg_lambda = 1e-3       # Coeficiente de regularización
n_train_data = 5000     # Cantidad de datos que se usarán para entrenar

nro_batchs = int(ytr.shape[0] / batch_size)  # Cantidad de batches

# inicializo las W de cada capa
W1 = np.random.normal(0, 1, (xtr.shape[1], nintermedia)) * 0.01  # 3073 x 100, meto el primer bias
W2 = np.random.normal(0, 1, (W1.shape[1] + 1, nclases)) * 0.1  # 101 x 10 : meto el segundo bias

# Se inicializan los vectores para guardar información para plotear
train_acc = np.zeros((n_epocas, 1))
train_loss = np.zeros((n_epocas, 1))
test_acc = np.zeros((n_epocas, 1))
test_loss = np.zeros((n_epocas, 1))
epoca = []

# Comienza el entrenamiento:
for i in range(n_epocas):
    indice = xtr.shape[0]
    indice = np.arange(indice)
    np.random.shuffle(indice)
    loss = 0
    for j in range(nro_batchs):
        # Forward
        x_batch = xtr[indice[(j * batch_size):((j + 1) * batch_size)], :]  # de batch_size x 3073
        y_batch = ytr[indice[(j * batch_size):((j + 1) * batch_size)]]  # de batch_size x 1

        yyy_tr = yy_tr[indice[(j * batch_size):((j + 1) * batch_size)], :]  # de batch_size x 10

        # Capa 1:
        Y1 = np.dot(x_batch, W1)  # esto es un vector de batch_size x 101
        S1 = sigmoide(Y1)
        S1_prima = np.hstack([np.ones((x_batch.shape[0], 1)), S1])  # esto es un vector de batch_size x 101

        # Capa 2
        S2 = np.dot(S1_prima, W2)  # es un vector de batch_size x 10
        y_test_estimado = np.dot(xt, W1)
        y_test_estimado_S1 = sigmoide(y_test_estimado)
        y_test_estimado_S1_prima = np.hstack([np.ones((xt.shape[0], 1)), y_test_estimado_S1])
        y_test_estimado_S2 = np.dot(y_test_estimado_S1_prima, W2)

        # Calculo la regularización
        reg1 = np.sum(W1 * W1)
        reg2 = np.sum(W2 * W2)
        reg = reg1 + reg2

        # Backward
        # Capa 1
        grad = gradCCE(S2, yyy_tr)  # gradiente de la Función costo
        grad = grad + 2 * (np.sum(W1) + np.sum(W2)) * reg_lambda / 2

        grad_W2 = np.dot(S1_prima.T, grad) # delta W2
        grad = np.dot(grad, W2.T)
        grad = grad[:, 1:]

        # Capa 2
        grad = grad_sigmoide(Y1) * grad # gradiente de la Función de activación
        grad_W1 = np.dot(x_batch.T, grad)

        # Actualización de los pesos
        W1 = W1 - learning_rate * (grad_W1 + reg * W1)
        W2 = W2 - learning_rate * (grad_W2 + reg * W2)

        # Cálculo de la loss y el accuracy
        train_loss[i] += CCE(S2, yyy_tr) + (reg1 + reg2) * 0.5
        train_acc[i] += accuracy(S2, y_batch)
        test_loss[i] += CCE(y_test_estimado_S2, yt) + (reg1 + reg2) * 0.5
        test_acc[i] += accuracy(y_test_estimado_S2, yt)

    epoca.append(i)

    train_acc[i] = train_acc[i]/nro_batchs
    train_loss[i] = train_loss[i]/nro_batchs
    test_acc[i] = test_acc[i] / nro_batchs
    test_loss[i] = test_loss[i]/nro_batchs

    print('Epoca: ' + str(i) + '  Accuracy: ' + str(train_acc[i]) + '     Loss: ' + str(test_loss[i]))


# plt.figure(1)
# plt.plot(epoca, train_acc,'r*')
# plt.plot(epoca, test_acc,'k*')
plt.plot(epoca, test_acc,'*')
# plt.legend(['Training data', 'Testing data'])
# plt.title('Evolución del accuracy - CIFAR10')
# plt.xlabel('Época')
# plt.ylabel('Accuracy')
# plt.savefig('Accuracy{}bz{}td{}.pdf'.format(n_epocas, batch_size, n_train_data))
# fig, ax1 = plt.subplots()
#
# # plt.plot(epoca, train_loss, 'r-*')
# # plt.plot(epoca, test_loss, 'k-*')
# plt.title('Evolución de la Loss - CIFAR10')
# ax1.set_xlabel('Época')
# ax1.set_ylabel('Loss', color='tab:red')
# ax1.tick_params(axis='y', labelcolor='r')
# # plt.legend(['Training data', 'Testing data'])
#
# ax1.plot(epoca, train_loss, color='tab:red',marker='*')
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'k'
# ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
# ax2.plot(epoca, test_loss, color=color, marker='*')
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('Loss_ep{}bz{}td{}.pdf'.format(n_epocas, batch_size, n_train_data))
# # plt.ylim(0,train_loss[1])
# plt.show(block=0)
# plt.pause(1)


###############   Esto sirve para plotear los 3 métodos juntos  ##############################

# plt.figure()
# plt.plot(epoca, train_acc,'r*')
# plt.plot(epoca, test_acc,'r*')
plt.legend(['MSE', 'SVM', 'CCE'])
plt.title('Comparación de la evolución del accuracy - CIFAR10')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.savefig('Accuracy{}bz{}td{}.pdf'.format(n_epocas, batch_size, n_train_data))

plt.show()