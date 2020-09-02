>>> from sklearn import datasets
>>> from sklearn.metrics import confusion matrix
  File "<stdin>", line 1
    from sklearn.metrics import confusion matrix
                                               ^
SyntaxError: invalid syntax
>>> from sklearn.metrics import confusion_matrix
>>> from sklearn.model_selection import train_test_split
>>> iris = datasets.load_iris()
>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib as plt
>>> from sklearn.metrics import accuracy_score
>>> class_name=['setosa','versicolor','virginica']
>>> X = iris.data
>>> y = iris.target
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
>>> np.unique(y_test)
array([0, 1, 2])
>>> np.unique(X_test)
array([0.1, 0.2, 0.3, 0.4, 1. , 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
       2. , 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3. , 3.1, 3.2, 3.3,
       3.4, 3.5, 3.6, 3.8, 3.9, 4. , 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8,
       4.9, 5. , 5.1, 5.2, 5.4, 5.5, 5.6, 5.7, 5.8, 6. , 6.1, 6.2, 6.3,
       6.4, 6.5, 6.7, 6.8, 7.3])
>>> from sklearn.tree import DesisionTreeClassifier
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ImportError: cannot import name 'DesisionTreeClassifier' from 'sklearn.tree' (D:\Anaconda\lib\site-packages\sklearn\tree\__init__.py)
>>> from sklearn.tree import DecisionTreeClassifier
>>> dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
>>> dtree_predictions = dtree_model.predict(X_test)
>>> cm = confusion_matrix(y_test, dtree_predictions)
>>> print(cm)
[[13  0  0]
 [ 0 15  1]
 [ 0  3  6]]
>>> from sklearn.metrics import classification_report
>>> report = classification_report(y_test, dtree_predictions)
>>> print(report)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        13
           1       0.83      0.94      0.88        16
           2       0.86      0.67      0.75         9

    accuracy                           0.89        38
   macro avg       0.90      0.87      0.88        38
weighted avg       0.90      0.89      0.89        38

>>>                                                                                      