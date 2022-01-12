#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        print('Vim ao linearmodel: init!')
        self.W = np.zeros((n_classes, n_features))
        print('Sai do linearmodel: init!')

    def update_weight(self, x_i, y_i, **kwargs):
        print('Vim ao linearmodel: udpate!')
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        print('Vim ao linearmodel: train_epoch!')
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)
        print('sai ao linearmodel: train_epoch!')

    def predict(self, X):#pega em y=sign(wT*x) e devolve os indices que correspondem aos maiores valores(1?)
        """X (n_examples x n_features)"""
        print('Vim ao linearmodel: predict!')
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        print("EU sou o predicted label:")
        print(predicted_labels)
        print('sai ao linearmodel: predict!')
        
        return predicted_labels

    def evaluate(self, X, y): 
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        print('Vim ao linearmodel: evaluate!')
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        print('Sai ao linearmodel: evaluate!')
        return n_correct / n_possible
#probabilidade da label ser 1

class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        print('Vim ao perceptron: update weigts!')
        y_hat = self.predict(x_i)
        if y_hat != y_i:
            self.W[y_i] = self.W[y_i] + x_i.T
            self.W[y_hat] = self.W[y_hat] - x_i.T
        
        print('Sai do perceptron update weights!')
        return self.W
       
        """
        # Q3.1a
        raise NotImplementedError
        """
    
class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
    
        logitscores = self.predict(x_i)
        
        exp = np.exp(logitscores)
        sum_exp=np.sum(np.exp(logitscores))
        y_hat = exp/sum_exp
       
        erro = y_hat - y_i#DUVIDA 1: com ou sem
        gradient = np.dot(erro, x_i.T)
        
        self.W += learning_rate * gradient#DUVIDA 2: +/-?
        return self.W
        

        
        '''
        # Q3.1b
        raise NotImplementedError
        '''

class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        #hidden_size=200
        self.W=np.random.normal(np.zeros((n_classes, n_features)))
        # Initialize an MLP with a single hidden layer.
        raise NotImplementedError

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        raise NotImplementedError

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        raise NotImplementedError


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        print('Passei aqui 1!')
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))
        print('Passei aqui 2! com %d'%i)
        
    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
