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
    """Sigmoid functions for a numpy array"""
    return(np.array([1 / (1 + (math.e ** (-xi))) for xi in x]))


def sigprime(x):
    """Derivative of sigmoid functions for a numpy array"""
    return(sigmoid(x) * (np.ones(len(x)) - sigmoid(x)))


def normalize(array):
    return(array / np.sqrt(np.sum(array ** 2)))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


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

    def __post_init__(self):
        # For each node in output layer, generate empty weights and biases
        self.weights = np.random.randn(self.out_size, self.in_size) * \
            np.sqrt(2 / self.in_size)
        self.biases = np.random.uniform(0, 1, self.out_size)
        self.X = []

    def __call__(self, x):
        """Function: z = Wx + b"""
        self.X = x
        
        # Multiply by vector or dot product depending on if input is matrix or vector
        self.z = np.matmul(self.weights, x) + (self.biases if (x.shape[1] == 1) else 0)
            
        # Apply sigmoid only if output is a vector
        self.layer_output = sigmoid(self.z) if (x.shape[1] == 1) else self.z

        return(self.layer_output)


class Net():
    def __init__(self):
        pass

    def __call__(self, x):
        """Get prediction from nueral net"""
        pass

    def backprop(self, pred, actual, alpha = 0.01):
        t = np.zeros(len(pred))
        t[actual] = 1

        for l, L in enumerate(self.layers):
            # Find the error at each layer
            D = self.delta(l, t)

            # Update each layer weights given the error at each layer
            L.weights -= np.outer(D, L.X) * alpha

            # Update layer biases
            L.biases -= D * alpha


    def delta(self, l, t):
        """Find delta between each layer(l) and the target value(t)"""
        # Derivative of sigmoid(z) -> σ'(z) = σ(z)(1 - σ(z))
        dA = sigprime(self.layers[l].z)

        if l == len(self.layers) - 1:
            pred = self.layers[l].layer_output
            G = (pred - t) / (pred * (np.ones(len(pred)) - pred))
            S = sigprime(self.L2.z)
            return np.multiply(G, S)

        # Get the weights at the next layer
        w = self.layers[l + 1].weights

        return np.multiply(np.dot(w.T, self.delta(l + 1, t)), dA)


def loss(pred, actual):
    """ Binary cross entropy loss.
    Function: −(𝑦log(𝑝)+(1−𝑦)log(1−𝑝))"""
    alpha = 0.01
    loss = 0
    if actual > 9:
        return 0

    t = np.zeros(len(pred))
    t[actual] = 1

    for i in range(len(t)):
        loss -= t[i] * math.log(pred[i]) + (1 - t[i]) * math.log(1 - pred[i])

    return(loss)


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
