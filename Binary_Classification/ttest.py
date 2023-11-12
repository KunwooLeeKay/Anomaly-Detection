
import Module_Preprocessing_New as prep

pp = prep.Preprocessing()

train_X, test_X, train_Y, test_Y = pp.B_Splitter()

from imblearn.over_sampling import SMOTE

train_X, train_Y = SMOTE().fit_resample(train_X, train_Y)

from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
standard_scaler.fit(train_X)

train_X = standard_scaler.transform(train_X)
test_X = standard_scaler.transform(test_X)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000).fit(train_X, train_Y)
pred = lr.predict(test_X)

from sklearn.metrics import classification_report

print(classification_report(test_Y, pred))