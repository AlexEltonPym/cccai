
import numpy as np
from sklearn import tree

userData = [0, 1, 2, 3, 4]


for uNum in userData:
    user = np.zeros(len(userData), dtype=np.int) # can yoou chain these v
    user[uNum] = 1
    print(user)

X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
print(clf.predict([[1, 1]]))