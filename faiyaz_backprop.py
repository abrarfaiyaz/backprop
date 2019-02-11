#!/usr/bin/env python3
import numpy as np
from io import StringIO

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "/root/Desktop/MLCourse/hw3/backprop_files/adult/" #TODO: if doing development somewhere other than the cycle server, change this to the directory where a7a.train, a7a.dev, and a7a.test are

#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y,0) #treat -1 as 0 instead, because sigmoid's range is 0-1
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray([ys],dtype=np.float32).T, np.asarray(xs,dtype=np.float32).reshape(len(xs),NUM_FEATURES,1) #returns a tuple, first is an array of labels, second is an array of feature vectors

def init_model(args):
    w1 = None
    w2 = None

    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1,len(w2))
    else:
        #TODO (optional): If you want, you can experiment with a different random initialization. As-is, each weight is uniformly sampled from [-0.5,0.5).
        w1 = np.random.rand(args.hidden_dim, NUM_FEATURES) #bias included in NUM_FEATURES
        w2 = np.random.rand(1, args.hidden_dim + 1) #add bias column

    #At this point, w1 has shape (hidden_dim, NUM_FEATURES) and w2 has shape (1, hidden_dim + 1). In both, the last column is the bias weights.
    #print(w1[:,0:3])
    #print(w2[:,0:2])
    args.hidden_dim = w1.shape[0]
    
    #TODO: Replace this with whatever you want to use to represent the network; you could use use a tuple of (w1,w2), make a class, etc.
    #model = {'W1':w1[:,0:123],'b1':w1[:,-1],'W2':w2[:,0:2],'b2':w2[:,-1]}
    #bw=np.zeros((1,124))
    #w1=np.concatenate((w1,bw),axis=0)
    model = {'W1':w1,'W2':w2}
    #raise NotImplementedError #TODO: delete this once you implement this function
    return model

def sigmoid(x):
    #Activation function 
    return 1/(1+np.exp(-x))

def sigmoidPrime(s):
    #derivative - sigmoid
    return s * (1 - s)

def forward(model, X):
    W1=model['W1']
    W2=model['W2']
    #forward propagation through network
    z = np.dot(X.T,W1.T) # dot product of X (input) and first set of 3x2 weights
    z2 = sigmoid(z) # activation function
    z2=np.concatenate((z2,[[1]]),axis=1)#bias appended    
    z3 = np.dot(z2, W2.T) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = sigmoid(z3) # final activation function
    return o

def backward(model, X, expected_output,args):
    W1=model['W1']
    W2=model['W2']
    #forward propagation through network
    z = np.dot(X.T,W1.T) # dot product of X (input) and first set of 3x2 weights
    z2 = sigmoid(z) # activation function
    z2=np.concatenate((z2,[[1]]),axis=1)#bias appended    
    z3 = np.dot(z2, W2.T) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = sigmoid(z3) # final activation function
	
    # backward propgate through the network
    o_error = args.lr*(expected_output - o) # error in output
    o_delta = o_error*sigmoidPrime(o) # applying derivative of sigmoid to error

    z2_error = o_delta.dot(W2) # z2 error: how much our hidden layer weights contributed to output error
    z2_delta = z2_error*sigmoidPrime(z2) # applying derivative of sigmoid to z2 error

    W1 += np.dot(z2_delta.T,X.T)[0:args.hidden_dim,:] # adjusting first set (input --> hidden) weights
    #print(W2.shape)
    W2 += np.dot(o_delta,z2) # adjusting second set (hidden --> output) weights
    model = {'W1':W1,'W2':W2}
    return model

def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args):
    #TODO: Implement training for the given model, respecting args
    for i in range(args.iterations):
        sum_e = 0
        for inputs,expected_output in zip(train_xs,train_ys):
            output = forward(model,inputs)
            model = backward(model,inputs,expected_output,args)
            sum_e +=(expected_output-output)**2
        if(i%10==0) and not args.nodev: #Accuracy is printed on development data
            #print('Loss : ',sum_e/i)
            acc=test_accuracy(model,dev_ys,dev_xs)
            print(acc)
        else:
            print('Loss : ',sum_e/i)
    if not args.nodev: 
        for i in range(args.iterations):
            sum_e = 0
            for inputs,expected_output in zip(dev_xs,dev_ys):
                output = forward(model,inputs)
                model = backward(model,inputs,expected_output,args)
                sum_e +=(expected_output-output)**2
            
    #raise NotImplementedError #TODO: delete this once you implement this function
    return model

def test_accuracy(model, test_ys, test_xs):
    accuracy = 0.0
    acc=0
    #TODO: Implement accuracy computation of given model on the test data
    for inputs,outputs in zip(test_xs,test_ys):
        out=forward(model,inputs)
        #print(out,outputs)
        if (outputs[0]==0 and out[0,0] < 0.5) or (outputs[0]==1 and out[0,0] >= 0.5) :
            acc+=1
    accuracy= np.float(acc/test_ys.size)
    #raise NotImplementedError #TODO: delete this once you implement this function
    return accuracy

def extract_weights(model):
    w1 = model['W1']
    w2 = model['W2']
    #TODO: Extract the two weight matrices from the model and return them (they should be the same type and shape as they were in init_model, but now they have been updated during training)
    #raise NotImplementedError #TODO: delete this once you implement this function
    return w1, w2

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Neural network with one hidden layer, trainable with backpropagation.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate to use for update in training loop.')

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1','W2'), type=str, help='Files to read weights from (in format produced by numpy.savetxt). First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')

    parser.add_argument('--print_weights', action='store_true', default=False, help='If provided, print final learned weights to stdout (used in autograding)')

    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')


    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.weights_files: iterable of str; if present, contains two fields, the first is the file to read the first layer's weights from, second is for the second weight matrix.
    args.hidden_dim: int; number of hidden layer units. If weights_files is provided, this argument should be ignored.
    args.train_file: str; file to load training data from.
    args.dev_file: str; file to load dev data from.
    args.test_file: str; file to load test data from.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)

    model = init_model(args)
    model = train_model(model, train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(model, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1,w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2,w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))

if __name__ == '__main__':
    main()
