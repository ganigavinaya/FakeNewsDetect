
from sklearn import model_selection
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
# $example off$
from sklearn.ensemble import AdaBoostClassifier



formatted = pd.read_csv('datafiles/FinalTrainData.csv')
print(formatted.shape)
df = formatted.fillna(formatted.mean())
# print(df.shape)

X = df.drop("Label", axis =1)

y = df["Label"]

# print(X.shape[1])
max_features = 30
num_trees = 500
# kfold = model_selection.KFold(n_splits=10)
rfc = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)

rfc.fit(X,y)
#results = model_selection.cross_val_score(rfc, X, y, cv=5)
#print(results)
#print(results.mean())

filename = 'rfc.joblib.pkl'
_ = joblib.dump(rfc, filename, compress=9)


# load_rfc = joblib.load('code/rfc.joblib.pkl')
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
adb = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
adb.fit(X,y)
#results = model_selection.cross_val_score(adb, X, y, cv=kfold)
#print(results.mean())



filename = 'adb.joblib.pkl'
_ = joblib.dump(adb, filename, compress=9)


# load_rfc = joblib.load('code/adb.joblib.pkl')