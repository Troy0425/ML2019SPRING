import numpy as np
import math
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
train = pd.read_csv(sys.argv[1], header=0)
train_x = pd.read_csv(sys.argv[3], header=0)
train_y = pd.read_csv(sys.argv[4], header=0)
test_x = pd.read_csv(sys.argv[5], header=0)
deletecol = ['?_workclass', '?_occupation', '?_native_country']
addname = ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']
def preprocess(data):
	#data = data.drop(deletecol, axis = 1)
	data = data.astype(np.float64)
	for k in addname:
		for i in range(1, 25):
			data['sin(%s)**%d' %(k, i)] = pd.Series(np.sin(data.loc[:,k]) ** i)
			data['cos(%s)**%d' %(k, i)] = pd.Series(np.cos(data.loc[:,k]) ** i)
			data['tan(%s)**%d' %(k, i)] = pd.Series(np.tan(data.loc[:,k]) ** i)
			data['arctan(%s)**%d' %(k, i)] = pd.Series(np.arctan(data.loc[:,k]) ** i)
		for i in range(2, 25):
			data['%s**%d' %(k, i)] = pd.Series(data.loc[:, k] ** i)
	data['bias'] = pd.Series(np.ones(data.shape[0], np.float64))
	return data

train_x = preprocess(train_x)
test_x = preprocess(test_x)
scaler = StandardScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)

#X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y.values.ravel(), test_size=0.2)

###gradient boosting classifier
clf = GradientBoostingClassifier(n_estimators = 200, max_depth = 4)
clf.fit(train_x, train_y.values.ravel())
predict = clf.predict(test_x)
#print("sk.gbc error:", clf.score(train_x, train_y))


###predict
fp = open(sys.argv[6], "w")
fp.write("id,label\n")
for i in range(test_x.shape[0]):
	fp.write("%d,%d\n" %(i + 1,predict[i]))
fp.close()
