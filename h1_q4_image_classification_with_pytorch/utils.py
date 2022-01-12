import os
import random

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import time


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fetch_classification_data(dataset="Fashion-MNIST", random_state=42):
    """
    Loads the dataset from openml, normalizes feature values (by dividing
    everything by 256), and saves to an npz file.

    dataset: the name of the dataset (accepted: "mnist_784", "Fashion-MNIST")
    """
    assert dataset in {"mnist_784", "Fashion-MNIST"}
    start_time = time.time()
    X, y = fetch_openml(dataset, version=1, return_X_y=True, as_frame=False)
    print("Downloaded data in {:.4f} seconds".format(time.time() - start_time))
    X /= 256  # normalize
    y = y.astype(int)  # fetch_openml loads it as a str
    train_dev_X, train_dev_y = X[:60000], y[:60000]
    train_X, dev_X, train_y, dev_y = train_test_split(
        train_dev_X, train_dev_y, train_size=50000, test_size=10000, random_state=random_state
    )
    test_X, test_y = X[60000:], y[60000:]
    np.savez_compressed(
        dataset + ".npz",
        Xtrain=train_X, ytrain=train_y,
        Xdev=dev_X, ydev=dev_y,
        Xtest=test_X, ytest=test_y
    )


def load_classification_data(bias=False):
    """
    Loads the preprocessed, featurized fashion-mnist dataset from
    Fashion-MNIST.npz, optionally adding a bias feature.
    """
    data = np.load('Fashion-MNIST.npz')
    train_X = data["Xtrain"]
    dev_X = data["Xdev"]
    test_X = data["Xtest"]
    if bias:
        train_X = np.hstack((train_X, np.ones((train_X.shape[0], 1))))
        dev_X = np.hstack((dev_X, np.ones((dev_X.shape[0], 1))))
        test_X = np.hstack((test_X, np.ones((test_X.shape[0], 1))))
    return {"train": (train_X, data["ytrain"]),
            "dev": (dev_X, data["ydev"]),
            "test": (test_X, data["ytest"])}


def load_regression_data(bias=False):
    """
    Loads the preprocessed, featurized Ames housing dataset from ames.npz.
    """
    data = np.load('ames.npz')
    train_X = data["Xtrain"]
    test_X = data["Xtest"]
    train_y = data["ytrain"].reshape(-1)
    test_y = data["ytest"].reshape(-1)
    if bias:
        train_X = np.hstack((train_X, np.ones((train_X.shape[0], 1))))
        test_X = np.hstack((test_X, np.ones((test_X.shape[0], 1))))
    return {"train": (train_X, train_y),
            "test": (test_X, test_y)}


class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        """
        data: the dict returned by utils.load_classification_data
        """
        train_X, train_y = data["train"]
        dev_X, dev_y = data["dev"]
        test_X, test_y = data["test"]

        self.X = torch.tensor(train_X, dtype=torch.float32)
        self.y = torch.tensor(train_y, dtype=torch.long)

        self.dev_X = torch.tensor(dev_X, dtype=torch.float32)
        self.dev_y = torch.tensor(dev_y, dtype=torch.long)

        self.test_X = torch.tensor(test_X, dtype=torch.float32)
        self.test_y = torch.tensor(test_y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
