import time
import math
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import tqdm
from dataclasses import dataclass


def sigmoid(x):
    """Sigmoid function for a numpy array"""
    return(np.array([1 / (1 + (math.e ** (-xi))) for xi in x]))


def sigprime(x):
    """Derivative of sigmoid function for a numpy array"""
    return(np.matmul(sigmoid(x), (np.ones(x.shape) - sigmoid(x)).T))


def relu(x):
    """ReLU function for a numpy array"""
    return(np.maximum(0, x))


def reluprime(x):
    """Derivative of ReLU function for a numpy array"""
    return(np.clip(x, a_min=0, a_max=1))


def normalize(array):
    return(array / np.sqrt(np.sum(array ** 2)))


#def softmax(x):
#    """Compute softmax values for each sets of scores in x."""
#    e_x = np.exp(x - np.max(x))
#    return e_x / e_x.sum(axis=0) # only difference


def softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class MnistDataLoader():
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return (len(self.images))

    def __getitem__(self, idx):
        return({
            "image":self.images[idx],
            "label":int(self.labels[idx]),
        })

    def rand_sample(self, n):
        r = random.randint(0, len(self.images))

        output = []
        for i in range(r, r + n):
            idx = (i % len(self.images))
            output.append({
                "image":self.images[idx],
                "label":int(self.labels[idx]),
            })
        return output


@dataclass
class LinearLayer():
    in_size: int
    out_size: int
    depth: int = 1
    
    def __post_init__(self, activation=relu, activation_p=reluprime, eps=1e-12):
        self.eps = eps
        
        # For each node in output layer, generate empty weights and biases
        self.activation = activation
        self.activation_p = activation_p
        
        # Handle position-wise FFN and give depth to linear layer(useful for conv)
        self.size = (self.out_size, self.in_size)
        if self.depth > 1:
            self.size += (self.depth,)
        
        # Init weights and biases
        self.weights = np.random.randn(*self.size) * np.sqrt(2 / self.in_size)
        self.biases = np.random.uniform(0, 1, (self.out_size, self.depth))
        self.X = []

    def __call__(self, x):
        """Function: z = Wx + b"""
        self.X = x
        
        # Multiply by vector or dot product depending on if input is matrix or vector
        ein_sum = "ijk,jk->ik" if (len(self.size) > 2) else "ik,kj->ij"
        self.z = np.einsum(ein_sum, self.weights, x) + self.biases
            
        # Apply sigmoid only if output is a vector
        self.layer_output = self.activation(self.z)

        return(self.layer_output)
    
    def delta(self, t):
        """Gets delta of layer for log loss"""
        pred = self.layer_output
        
        # Derivative of layer given output from layer call
        G = (pred - t) / ((pred * np.ones(pred.shape)) - pred + self.eps) 
        S = self.activation_p(self.z)
        
        x = np.multiply(G, S)
        print(x.shape)
        
        # If linear layer is a tensor
        if len(self.size) > 2:
            return np.einsum("kj,ik->ij", G.T, S)
        
        return np.multiply(G, S)
    

class Net():
    """Neural net module"""
    
    def __init__(self, activation="ReLU"):
        if (activation == "ReLU"):
            self.activation = relu
            self.activation_p = reluprime
            
        elif (activation == "Sigmoid"):
            self.activation = sigmoid
            self.activation_p = sigprime
            
    def __call__(self, x):
        """Get prediction from nueral net"""
        pass

    def backprop(self, pred, t, alpha = 0.01):
        for index in range(pred.shape[0] - 1):
            for l, L in enumerate(self.layers):
                # Find the error at each layer
                D = self.delta(l, t, index)

                # Update each layer weights given the error at each layer
                L.weights[:,:,index] -= np.outer(D[:,index], L.X[:,index]) * alpha

                # Update layer biases
                #L.biases -= np.sum(D, axis=1) * alpha


    def delta(self, l, t, index):
        """Find delta between each layer(l) and the target value(t)"""
        # Derivative of sigmoid(z) -> σ'(z) = σ(z)(1 - σ(z))
        dA = self.activation(self.layers[l].z)

        if l == len(self.layers) - 1:
            pred = self.layers[l].layer_output[:,index]
            print("pred_shape", pred.shape)
            print("tshape", t.shape)
            
            G = (pred - t[index:,]) / ((pred * np.ones(pred.shape)) - pred)
            S = self.activation_p(self.layers[l].z)
            return np.multiply(G.T, S)

        # Get the weights at the next layer
        w = self.layers[l + 1].weights[:,:,index]
        
        return np.multiply(np.dot(w.T, self.delta(l + 1, t, index)), dA)
    
    
def cross_entropy(x, t):
    """ Binary cross entropy loss.
    Function: −(𝑦log(𝑝)+(1−𝑦)log(1−𝑝))"""
    loss = 0
    epsilon=1e-12
    x = np.clip(x, epsilon, 1. - epsilon)
    return(np.sum(t * np.log(x+1e-9)) / (x.shape[0] + epsilon))

    
def loss(pred, target, epsilon=1e-12):
    """Loss summation function for """
    return(np.apply_along_axis(cross_entropy, 1, pred,  target).sum())


def train_model(model, train_data, val_data, num_epochs=20):
    stats = {
        "accuracy":{"Train":[],"Val":[]},
        "loss":{"Train":[], "Val":[]}
    }

    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        for phase in ["Train", "Val"]:
            dataset = train_data if phase == "Train" else val_data
            correct = 0
            running_loss = 0

            # Iterate through train and val datasets
            for image in tqdm.tqdm(dataset, desc=phase):
                result = model.forward(image["image"])

                running_loss += loss(result, image["label"])
                alpha = 0.0001 if epoch < 3 else 0.001

                # Only gradient descent if training
                if phase == "Train":
                    model.backprop(result, image["label"], alpha)

                if image["label"] == np.argmax(result):
                    correct += 1

            # Training statistics
            stats["accuracy"][phase].append(acc := correct / len(dataset))
            stats["loss"][phase].append(running_loss)

            print("{} Accuracy:".format(phase), acc)
            print("{} Loss:".format(phase), running_loss)
        print()

    plot_train_data(stats)

    return model


def plot_train_data(stats):
    """Plots all data from model training"""
    fig = plt.figure(1)
    for i, stat in enumerate(stats):
        ax = fig.add_subplot(2, 1, i + 1)
        ax.set_ylabel(stat)

        for phase in stats[stat]:
            ax.plot(stats[stat][phase], label="{} {}".format(phase, stat))
            ax.legend(loc="upper left")

    plt.show()
