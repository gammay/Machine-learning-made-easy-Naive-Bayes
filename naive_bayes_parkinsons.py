import pandas as pd

# Feature columns we use
x_rows=['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)',
        'MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP']
# x_rows=['MDVP:Fo(Hz)']
y_rows=['status']

# Train

# Read train data
train_data = pd.read_csv('parkinsons/Data_Parkinsons_TRAIN.csv')
train_x = train_data[x_rows]
train_y = train_data[y_rows]
print("train_x:\n", train_x)
print("train_y:\n", train_y)

# Load sklearn Gaussian Naive Bayes and fit
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB() 
gnb.fit(train_x, train_y) 

# Prediction on train data
predict_train = gnb.predict(train_x)
print('Prediction on train data:', predict_train) 

# Accuray score on train data
from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(train_y, predict_train)
print('Accuracy score on train data:', accuracy_train)

# Test

# Read test data
test_data = pd.read_csv('parkinsons/Data_Parkinsons_TEST.csv')
test_x = test_data[x_rows]
test_y = test_data[y_rows]

# Prediction on test data
predict_test = gnb.predict(test_x)
print('Prediction on test data:', predict_test) 

# Accuracy Score on test data
accuracy_test = accuracy_score(test_y, predict_test)
print('Accuracy score on test data:', accuracy_train)

# Prediction on unknown data
# predict_unknown = gnb.predict([[150,160,70,0,0,0,0,0]])
predict_unknown = gnb.predict([[240,250,230,0,0,0,0,0]])
print('Prediction on unknown data:', predict_unknown) 
