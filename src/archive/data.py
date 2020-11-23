import tensorflow as tf
import numpy as np
import pandas as pd
import os


class DatasetCreator:
    @classmethod
    def get_data(cls, filename):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'datasets', filename)
        data = pd.read_feather(path)
        return data.copy()

    @classmethod
    def split_data(cls, data, ratio):
        # Set seed
        np.random.seed(42)
        # Generate random indices
        indices = np.random.permutation(len(data))
        # Calculate how many entries the test data will have
        test_size = int(len(data)*ratio)
        # Get the test indices from the randomly generated indices
        test_indices = indices[:test_size]
        # Get the train indices from the randomly generated indices
        train_indices = indices[test_size:]
        # Return the data corresponding to the indices
        return data.iloc[test_indices], data.iloc[train_indices]

    @classmethod
    def create_dataset(cls, data, labels, parameters):
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))\
                  .batch(parameters['batch_size'])\
                  .shuffle(data.shape[0])\
                  .prefetch(parameters['batch_size']*4)
        return dataset

    def __init__(self, filename, parameters, reshape=False, cache=True):
        # Read data
        data = DatasetCreator.get_data(filename)
        # Split data
        test_set, train_set = DatasetCreator.split_data(data, 0.2)
        # Split into labels and features
        test_labels = np.round(test_set.iloc[:, 0].to_numpy(), 3)
        test_data = np.round(test_set.iloc[:, 1:].to_numpy(), 3)
        train_labels = np.round([train_set.iloc[:, 0].to_numpy()], 3).T
        train_data = np.round(train_set.iloc[:, 1:].to_numpy(), 3)
        # Reshape
        if reshape:
            st_sz = parameters['stencil_size']
            test_data = np.reshape(test_data, (test_data.shape[0], st_sz[0], st_sz[1], 1))
            train_data = np.reshape(train_data, (train_data.shape[0], st_sz[0], st_sz[1], 1))

        self.train_dataset = DatasetCreator.create_dataset(train_data, train_labels, parameters)
        self.test_dataset = DatasetCreator.create_dataset(test_data, test_labels, parameters)

        # For speed:
        if cache:
            self.train_dataset = self.train_dataset.cache()
            self.test_dataset = self.test_dataset.cache()

    def get_train(self):
        return self.train_dataset

    def get_test(self):
        return self.test_dataset
