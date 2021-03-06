# -*- coding: utf-8 -*-
"""Eksperymenty magisterka.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16UyR9ssNXe_5ymn6iXkFgPOErGBhYrDU
"""

!pip install data-complexity

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier



from DDAG import DDAG
from ADAG import ADAG
from C_DDAG import C_DDAG

import numpy as np
from dcm import dcm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.base import clone

clfs = {
#    'SVM': SVC(),
#    'CART': DecisionTreeClassifier(),
#    'GNB': GaussianNB(),
#    'LOG': LogisticRegression(),
#    'KNN': KNeighborsClassifier(),
    'MLP': MLPClassifier(),
    'DDAG': DDAG(),
    'ADAG': ADAG(),
    'F1DAG': C_DDAG(), 
    'N1DAG': C_DDAG()  
}

datasets = ['automobile', 'balance', 'car', 'cleveland', 'contraceptive', 'dermatology', 'ecoli', 'flare', 'glass', 'hayes-roth', 'led7digit', 'lymphography', 'newthyroid', 'page-blocks', 'thyroid', 'vehicle', 'wine', 'winequality-red', 'yeast', 'zoo']
#datasets = ['lymphography', 'yeast', 'cleveland', 'vehicle', 'flare']

n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

for data_id, dataset in enumerate(datasets):
    print(dataset)
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):   
                   
        for clf_id, clf_name in enumerate(clfs):       
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = balanced_accuracy_score(y[test], y_pred)
            print(clf_name)
np.save('results', scores)