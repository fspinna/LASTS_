#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:16:29 2020

@author: francesco
"""

from pyts.datasets import make_cylinder_bell_funnel, load_gunpoint, load_coffee
from tslearn.generators import random_walk_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler  # doctest: +NORMALIZE_WHITESPACE
import warnings


def build_cbf(n_samples=600, random_state=0, verbose=True):
    X_all, y_all = make_cylinder_bell_funnel(n_samples=n_samples, random_state=random_state)
    X_all = X_all[:, :, np.newaxis]

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all,
        y_all,
        test_size=0.3,
        stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp,
        y_exp,
        test_size=0.2,
        stratify=y_exp,
        random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train, y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_ts_syege(path="./datasets/ts_syege/", random_state=0, verbose=True):
    X_all = np.load(path + "ts_syege01.npy")
    X_all = X_all[:, :, np.newaxis]

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        # print("y SHAPE: ", y_all.shape)
        # unique, counts = np.unique(y_all, return_counts=True)
        # print("\nCLASSES BALANCE")
        # for i, label in enumerate(unique):
            # print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp = train_test_split(
        X_all,
        test_size=0.3,
        random_state=random_state
    )

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test = train_test_split(
        X_train,
        test_size=0.2,
        random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val = train_test_split(
        X_train,
        test_size=0.2,
        random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test = train_test_split(
        X_exp,
        test_size=0.2,
        random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val = train_test_split(
        X_exp_train,
        test_size=0.2,
        random_state=random_state
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, None, X_val, None, X_test, None, X_exp_train,
            None, X_exp_val, None, X_exp_test, None)


def build_multivariate_cbf(n_samples=600, n_features=3, random_state=0, verbose=True):
    X_all = [[], [], []]
    y_all = []
    for i in range(n_features):
        X, y = make_cylinder_bell_funnel(n_samples=n_samples, random_state=random_state + i)
        X = X[:, :, np.newaxis]
        for label in range(3):
            X_all[label].append(X[np.nonzero(y == label)])
    for i in range(len(X_all)):
        X_all[i] = np.concatenate(X_all[i], axis=2)
    for label in range(3):
        y_all.extend(label for i in range(len(X_all[label])))
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.array(y_all)

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all,
        y_all,
        test_size=0.3,
        stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp,
        y_exp,
        test_size=0.2,
        stratify=y_exp,
        random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train, y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_synth(path="./datasets/synth04/", verbose=True, random_state=0):
    X_all = np.load(path + "X.npy")
    y_all = np.load(path + "y.npy").ravel()

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_exp, y_train, y_exp = train_test_split(
        X_all,
        y_all,
        test_size=0.3,
        stratify=y_all, random_state=random_state
    )

    # BLACKBOX TRAIN/TEST SETS SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # BLACKBOX TRAIN/VALIDATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # EXPLANATION TRAIN/TEST SETS SPLIT
    X_exp_train, X_exp_test, y_exp_train, y_exp_test = train_test_split(
        X_exp,
        y_exp,
        test_size=0.2,
        stratify=y_exp,
        random_state=random_state
    )

    # EXPLANATION TRAIN/VALIDATION SETS SPLIT
    X_exp_train, X_exp_val, y_exp_train, y_exp_val = train_test_split(
        X_exp_train, y_exp_train,
        test_size=0.2,
        stratify=y_exp_train,
        random_state=random_state
    )

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_gunpoint(random_state=0, verbose=True, label_encoder=True):
    X_all, X_test, y_all, y_test = load_gunpoint(return_X_y=True)
    X_all = X_all[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]
    if label_encoder:
        le = LabelEncoder()
        le.fit(y_all)
        y_all = le.transform(y_all)
        y_test = le.transform(y_test)

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        stratify=y_all, random_state=random_state
    )

    X_exp_train = X_train.copy()
    y_exp_train = y_train.copy()
    X_exp_val = X_val.copy()
    y_exp_val = y_val.copy()
    X_exp_test = X_test.copy()
    y_exp_test = y_test.copy()

    warnings.warn("Blackbox and Explanation sets are the same")

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)
        print("\nBlackbox and Explanation sets are the same!")

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_coffee(random_state=0, verbose=True, label_encoder=True):
    X_train, X_test, y_train, y_test = load_coffee(return_X_y=True)
    X_train = X_train[:, :, np.newaxis]
    X_test = X_test[:, :, np.newaxis]
    if label_encoder:
        le = LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_train.shape)
        print("y SHAPE: ", y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    X_exp_train = X_train.copy()
    y_exp_train = y_train.copy()
    X_exp_val = X_val = np.array(list())
    y_exp_val = y_val = np.array(list())
    X_exp_test = X_test.copy()
    y_exp_test = y_test.copy()

    warnings.warn("The validation sets are empty, use cross-validation to evaluate models")
    warnings.warn("Blackbox and Explanation sets are the same")

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


def build_ecg200(path="./datasets/ECG200/", random_state=0, verbose=True, label_encoder=True):
    X_all = pd.read_csv(path + "ECG200_TRAIN.txt", sep="\s+", header=None)
    y_all = np.array(X_all[0])
    X_all = np.array(X_all.drop([0], axis=1))
    X_all = X_all[:, :, np.newaxis]

    X_test = pd.read_csv(path + "ECG200_TEST.txt", sep="\s+", header=None)
    y_test = np.array(X_test[0])
    X_test = np.array(X_test.drop([0], axis=1))
    X_test = X_test[:, :, np.newaxis]

    if label_encoder:
        le = LabelEncoder()
        le.fit(y_all)
        y_all = le.transform(y_all)
        y_test = le.transform(y_test)

    if verbose:
        print("DATASET INFO:")
        print("X SHAPE: ", X_all.shape)
        print("y SHAPE: ", y_all.shape)
        unique, counts = np.unique(y_all, return_counts=True)
        print("\nCLASSES BALANCE")
        for i, label in enumerate(unique):
            print(label, ": ", round(counts[i] / sum(counts), 2))

    # BLACKBOX/EXPLANATION SETS SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        stratify=y_all, random_state=random_state
    )

    X_exp_train = X_train.copy()
    y_exp_train = y_train.copy()
    X_exp_val = X_val.copy()
    y_exp_val = y_val.copy()
    X_exp_test = X_test.copy()
    y_exp_test = y_test.copy()

    warnings.warn("Blackbox and Explanation sets are the same")

    if verbose:
        print("\nSHAPES:")
        print("BLACKBOX TRAINING SET: ", X_train.shape)
        print("BLACKBOX VALIDATION SET: ", X_val.shape)
        print("BLACKBOX TEST SET: ", X_test.shape)
        print("EXPLANATION TRAINING SET: ", X_exp_train.shape)
        print("EXPLANATION VALIDATION SET: ", X_exp_val.shape)
        print("EXPLANATION TEST SET: ", X_exp_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
            y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test)


if __name__ == "__main__":
    (X_train, y_train, X_val, y_val, X_test, y_test, X_exp_train,
     y_exp_train, X_exp_val, y_exp_val, X_exp_test, y_exp_test) = build_synth(path="./datasets/synth04/")
