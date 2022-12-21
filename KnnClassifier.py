import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p
        self.train_set = []


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """

        for i in range(len(X)): #append list of each point with x, y labels to the list of points.
            list_of_x_y=[X[i]]
            list_of_x_y.append(y[i])
            self.train_set.append(list_of_x_y)

        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        prediction = []
        np_train_set = np.array(self.train_set, dtype=object)

        for test in X:  # Going over the test points
            Minkowski_dist_array = np.zeros(shape=[len(np_train_set), 2])

            for t in range(len(np_train_set)):  # Going over the rows of the train set
                temp_dis = 0
                train_set_temp = np_train_set[t][0]
                for i in range(len(test)):  # Calculate the minkowski distance
                    temp_dis += (pow(abs(train_set_temp[i] - test[i]), self.p))
                Minkowski_dist = pow(temp_dis, 1 / self.p)
                Minkowski_dist_array[t, 0] = Minkowski_dist         # Put the minkowski distance in the array
                Minkowski_dist_array[t, 1] = np_train_set[t, -1]    # Put the label of the train point in the array

            Minkowski_dist_array = Minkowski_dist_array[Minkowski_dist_array[:, 1].argsort()]   # Sort by label
            Minkowski_dist_array = Minkowski_dist_array[Minkowski_dist_array[:, 0].argsort()]   # Sort by distance
            k_neighbors = Minkowski_dist_array[:self.k]     # Take the K nearest neighbors

            labels = []
            for neighbor in k_neighbors:    # Create array of the labels of the k nearest neighbors
                labels.append(neighbor[1])

            labels = np.bincount(labels)        # Count the most frequent label
            max_frec = max(labels)
            
            max_labels = []
            for h in range(len(labels)):        # Crate array with the most frequent labels of the K neighbors
                if labels[h] == max_frec:
                    max_labels.append(h)
                    
            if len(max_labels) == 1:        # If we have one label, append it to the predicted array
                prediction.append(max_labels[0])

            else:       # If we have tie - more than one max frequent label
                max_labels_neighbors = []
                for i in range(len(k_neighbors)):      # Creat array with the neighbors that have the most frequent label
                    for j in range(len(max_labels)):
                        if max_labels[j] == k_neighbors[i, 1]:
                            max_labels_neighbors.append(k_neighbors[i])

                np_max_labels_neighbors = np.array(max_labels_neighbors)
                min_dist = min(np_max_labels_neighbors[:, 0])       # Find min distance

                array_min_dists = []
                for i in range(len(np_max_labels_neighbors)):       # Create array of all neighbors with min distance
                    if np_max_labels_neighbors[i, 0] == min_dist:
                        array_min_dists.append(np_max_labels_neighbors[i])

                np_array_min_dists = np.array(array_min_dists)
                if len(np_array_min_dists) == 1:            # If there is one neighbor with min distance than append it to the predicted array
                    prediction.append(np_array_min_dists[0, 1])

                else:
                    min_label = min(np_array_min_dists[:, 1])       # There are more than one neighbor with min distance, find the min lexicographic label
                    prediction.append(min_label)

        return np.array(prediction)


def main():

    print("*" * 20)
    print("Started KnnClassifier.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == "__main__":
    main()

