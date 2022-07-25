#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
#ADJUSTMENTS MADE: import Dataframe to use the iloc function from pandas
from pandas import DataFrame
#END OF ADJUSTMENTS IN THIS PART
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
from pandas import to_numeric
from pandas import set_option
from scipy.io import loadmat
from numpy import reshape, empty, float32, int64, zeros
from PIL import Image, ImageDraw
set_option("display.max_rows", None, "display.max_columns", None)
#### Libraries
# Standard library
import time
import numpy as np
import matplotlib.pyplot as plt
import theano
import csv
import ipdb

# Third-party libraries
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
from theano.tensor import shared_randomstreams


# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

val_acc = list()
test_acc = list()

#### Constants
GPU = True
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False.")
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print("Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True.")

def load_data_shared1(filename="mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    print("training_data[0][0].shape",training_data[0][0].shape)
    print("training_data[1][0].shape",training_data[1][0])
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]


##PARTE VICTOR >>> função load_data_shared adaptada
def load_data_shared(samples="datasetEEG_extended_topo.mat", labels="datasetEEG_label_extended_topo.csv"):

    dic1 = loadmat(samples)
    rows = dic1['dataset2'][0][0].shape[0]
    cols = dic1['dataset2'][0][0].shape[1]
    chans = dic1['dataset2'][0][0].shape[2]
    dataset_shape = dic1['dataset2'].shape[0]

    def define_matrices(dataset_shape, rows, cols, chans):
        #samples_fft = empty((dataset_shape,rows*cols*chans),dtype=float32)
        samples_fft = empty((dataset_shape,2700),dtype=float32)
        print("1 samples_fft.shape",samples_fft.shape)
        return samples_fft

    def generate_dataset_samples(dic1,dataset_shape):
        for i in range(0,dataset_shape):
            #pil_fig = Image.fromarray(dic1['dataset2_fft'][i][0])
            pil_fig = Image.fromarray(dic1['dataset2'][i][0])
            resized = pil_fig.resize((30,30), Image.ANTIALIAS)
            #d1 = reshape(dic1['dataset2_fft'][i][0],(dic1['dataset2_fft'][i][0].shape[0]*dic1['dataset2_fft'][i][0].shape[1]*dic1['dataset2_fft'][i][0].shape[2]))
            d1 = reshape(resized,(resized.size[0]*resized.size[1]*3))
            #d1 = reshape(pil_fig,(pil_fig.size[0]*pil_fig.size[1]*3))
            norm_d1 = (d1 - d1.min(axis=0))/(d1.max(axis=0) - d1.min(axis=0))  #normalization
            norm_d1 = float32(norm_d1)
            samples_fft[i] = norm_d1
        print("2 samples_fft.shape",samples_fft.shape)
        return samples_fft


    def generate_dataset_labels(labels):
        labelling = []

        with open(labels,"r") as file:
            labels = csv.reader(file)
            cont = 0

            for e in labels:
                conv_num = int(e.pop())
                labelling.append(conv_num)
                cont+=1 
            labelling = int64(labelling)
        return labelling

    def splitting(samples_fft, labelling):
        train_size = 70
        test_size = 30
        X_tr_val, X_test, y_tr_val, y_test = train_test_split(samples_fft, labelling, train_size=train_size/100, random_state=666)
        X_tr, X_val, y_tr, y_val = train_test_split(X_tr_val, y_tr_val, test_size=test_size/100, random_state=666)
        return X_tr, y_tr, X_val, y_val, X_test, y_test

    samples_fft = define_matrices(dataset_shape, rows, cols, chans)
    samples_fft = generate_dataset_samples(dic1,dataset_shape)
    labelling = generate_dataset_labels(labels)
    X_tr, y_tr, X_val, y_val, X_test, y_test = splitting(samples_fft, labelling)

    training_data = (X_tr, y_tr)
    validation_data = (X_val, y_val)
    test_data = (X_test, y_test)
    input_size = training_data[0][0].shape[0]


    def shared(data):
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [input_size,shared(training_data), shared(validation_data), shared(test_data)]


#### Main class used to construct and train networks
class Network3(object):

    def __init__(self, layers, mini_batch_size):

        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)): # xrange() was renamed to range() in Python 3.
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):

        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = int(size(training_data)/mini_batch_size)
        num_validation_batches = int(size(validation_data)/mini_batch_size)
        num_test_batches = int(size(test_data)/mini_batch_size)

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        # Do the actual training

        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                our_zero = time.time()
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                    GetSecs = time.time() - our_zero
                    print("Execution mini-batch training time:",GetSecs)
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    val_acc.append(validation_accuracy)
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in range(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
                            test_acc.append(test_accuracy) 
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))
        return test_acc, val_acc


#### Define layer types

class ConvPoolLayer(object):

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = pool_2d(
            input=conv_out, ws=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    print("Funcionou!")
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)

def plot(v_acc, t_acc):
	# plot learning curves
	pyplot.xlabel('Época')
	pyplot.ylabel('Acurácia')
	pyplot.plot(t_acc, label='acuracia_teste', color='green')
	pyplot.plot(v_acc, label='acuracia_validacao', color='magenta')
	pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	pyplot.savefig("CNNTest2NonFFT.png",format='png', dpi=1600)


def main():
   
    input_size, training_data, validation_data, test_data = load_data_shared()    
    #training_data, validation_data, test_data = load_data_shared1()

    mini_batch_size = 32
    
    net = Network3([
        FullyConnectedLayer(n_in=input_size,n_out=15),
        FullyConnectedLayer(n_in=15,n_out=10),
        #FullyConnectedLayer(n_in=784, n_out=100),
        
        SoftmaxLayer(n_in=10, n_out=2)], mini_batch_size)
    net.SGD(training_data, 50, mini_batch_size, 0.1, validation_data, test_data)


if __name__ == "__main__":
    main()
