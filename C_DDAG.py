from pandas.core.accessor import delegate_names
from numpy.lib.function_base import delete
from dcm import dcm
#####################################

import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import clone
from copy import deepcopy
from scipy.stats import mode
from itertools import combinations

class C_DDAG(BaseEstimator, ClassifierMixin):
    """
    Implementacja DDAG wykorzystującego metryki złożoności problemu

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

        queue = np.unique(np.sort(self.y_))
        data = np.concatenate((self.X_, self.y_.reshape(self.y_.shape[0],1)), axis=1)
        comb = combinations(queue, 2)   

        self.models_list = {}
        self.complexity = {}

        for problem in list(comb):                 

          new_x = np.array([x[:-1] for x in data if x[-1] in problem])
          new_y = np.array([x[-1] for x in data if x[-1] in problem])
          
          N1 = dcm.N1(new_x, new_y)
          model = SVC().fit(new_x, new_y)
          
          #F1 = dcm.F1(new_x, new_y)

          self.models_list[problem] = model
          self.complexity[problem] = N1
          #self.complexity[problem] = F1

        print(self.complexity)
        return self

    def predict(self, X):        
        y_pred = np.zeros(X.shape[0])
        for i, x_query in enumerate(X):
          queue = np.unique(np.sort(self.y_))          
          np.random.shuffle(queue)         
          while len(queue) > 1:

            min = min(self.complexity.keys(), key=(lambda k: self.complexity[k]))
                        
                      
            problem = (min[0], min[1])

            if problem in self.models_list.keys(): model = self.models_list.get(problem)
            else: model = self.models_list.get(problem[::-1])
          
            pred = model.predict(x_query.reshape(1, -1))
            
            delete = [min[0], min[1]]
            idx  =  delete.index(int(pred[0]))    
            delete.pop(idx)

            queue = np.delete(queue, np.where(queue == delete[0]))
                      
          y_pred[i] = queue[0]
          
        return y_pred



#####################################