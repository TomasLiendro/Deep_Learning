import numpy as np


class Accuracy:
	def __call__(self, y_score=None, y_real=None, problem_name='XOR'):
		if problem_name == 'XOR':
			# acc = np.zeros((y_score.shape[0], 1))
			y_score[y_score > 0.9] = 1
			y_score[y_score < -0.9] = -1
			acc = np.mean(y_score == y_real) * 100
			return acc
		if problem_name == 'CIFAR10':
			acc = np.zeros((y_score.shape[0], 1))
			Y_pred = np.argmax(y_score, axis=1)
			yreal = np.argmax(y_real, axis=1)
			Y_pred = Y_pred[:, np.newaxis]
			b = Y_pred.T == yreal[:, np.newaxis].T
			acc[b.T] = 1
			acc_sum = np.sum(acc, axis=0) * 100
			y = acc_sum / y_score.shape[0]
			return y


class MSE:
	def __call__(self, y_score, y_real):
		mse = np.mean(np.sum((y_score - y_real) ** 2, axis=1))
		return mse
