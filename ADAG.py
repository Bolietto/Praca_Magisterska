from pandas.core.accessor import delegate_names
from numpy.lib.function_base import delete
#####################################

import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import clone
from copy import deepcopy
from scipy.stats import mode
from itertools import combinations

class ADAG(BaseEstimator, ClassifierMixin):
    """
    Implementacja ADAG, czyli adaptywnej metody opartej o strukturę DAG

    Parameters
    ----------
     base_estimator : opis

     iter : int (Default = 15)
        opis

     optimized : bool (Default = False)
        opis

     random_state : (chyba do wywalenia)
    
    """

    def __init__(self, random_state = None):
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X, y      

        queue = np.unique(self.y_)
        data = np.concatenate((self.X_, self.y_.reshape(self.y_.shape[0],1)), axis=1)
        comb = combinations(queue, 2)   

        self.models_list = {}

        for problem in list(comb):                 

          new_x = [x[:-1] for x in data if x[-1] in problem]
          new_y = [x[-1] for x in data if x[-1] in problem]

          model = SVC().fit(new_x, new_y)

          self.models_list[problem] = model

        return self

    def predict(self, X): 

        check_is_fitted(self)
        X = check_array(X)

        y_pred = np.zeros(X.shape[0])
        for i, x_query in enumerate(X):
          
          queue = np.unique(self.y_)
          splitedSize = 2
          
          while(len(queue) > 1):
            queue_splited = [queue[x:x+splitedSize] for x in range(0, len(queue), splitedSize)]

            for duel in queue_splited:
                            
              if len(duel) < 2: continue
              
              problem = (duel[0], duel[1])              

              if problem in self.models_list.keys(): model = self.models_list.get(problem)
              else: model = self.models_list.get(problem[::-1])
              pred = model.predict(x_query.reshape(1, -1))

              delete = [duel[0], duel[1]]

              idx  =  delete.index(int(pred[0]))    
              delete.pop(idx)

              queue = np.delete(queue, np.where(queue == delete[0]))

          y_pred[i] = queue[0]
          
        return y_pred



#####################################