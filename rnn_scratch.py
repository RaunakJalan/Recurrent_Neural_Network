import numpy as np

def bin2int(bin_list):
	bin_str = ''
	for k in bin_list:
		bin_str += str(int(k))

	return int(bin_str,2)


def dataset(num):
	bin_len = 8
	X = np.zeros((num,bin_len))
	Y = np.zeros((num))

	for i in range(num):
		X[i] = np.round(np.random.rand(bin_len)).astype(int)
		Y[i] = bin2int(X[i])

	return X,Y


num_samples = 20
num_samples_test = 5
train_X,train_Y = dataset(num_samples)
test_X,test_Y = dataset(num_samples_test)

# print(train_X)
# print(train_Y)


class RNN:
	def __init__(self):
		#defining some hyperparameters
		#first is input weight and second is recurrent weight
		self.W = [1,1]
		self.W_delta = [0.001,0.001]
		self.W_sign = [0,0]
		self.eta_p = 1.2
		self.eta_n = 0.5


	def state(self,xk,sk):
		#xk = input,sk = previous state
		return xk*self.W[0] + sk*self.W[1]

	def forward_state(self,X):
		S = np.zeros((X.shape[0],X.shape[1]+1))
		for k in range(0,X.shape[1]):
			next_state = self.state(X[:,k],S[:,k])
			S[:,k+1] = next_state

		return S

	def output_gradient(self,guess,real):
		return 2*(guess-real)/num_samples

	def backward_gradient(self,X,S,grad_out):
		grad_over_time = np.zeros((X.shape[0],X.shape[1]+1))
		grad_over_time[:,-1] = grad_out

		wx_grad = 0
		wr_grad = 0

		for k in range(X.shape[1],0,-1):
			wx_grad += np.sum(grad_over_time[:,k] * S[:,k-1])
			wr_grad += np.sum(grad_over_time[:,k] * S[:,k-1])

			grad_over_time[:,k-1] = grad_over_time[:,k] * self.W[1]

		return (wr_grad,wr_grad),grad_over_time

	def update_rprop(self,X,Y,W_prev_sign,W_delta,batch):
		S = self.forward_state(X)
		grad_out = self.output_gradient(S[:,-1],Y)

		if batch%1000 == 0:
			print("Loss: {:0.08f}".format(np.mean(grad_out)))

		W_grads, _ = self.backward_gradient(X,S,grad_out)
		self.W_sign = np.sign(W_grads)

		for i, _ in enumerate(self.W):
			if self.W_sign[i] == W_prev_sign[i]:
				W_delta[i] *= self.eta_p
			else:
				W_delta[i] *= self.eta_n

			self.W_delta = W_delta

	def train(self,X,Y,training_steps):
		for step in range(training_steps):
			self.update_rprop(X,Y,self.W_sign,self.W_delta,step)

			for i, _ in enumerate(self.W):
				self.W[i] -= self.W_sign[i] * self.W_delta[i]



rnn = RNN()
print("weights: \t\t",rnn.W)

rnn.train(train_X,train_Y,22000)

print("weights: \t\t",rnn.W)

print("Real:\t\t",test_Y)

y = rnn.forward_state(test_X)[:,-1]
print("Predicted: \t",np.round(y))








