import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RandomForest

data = pd.read_csv("Grand-slams-men-2013.csv", encoding='Latin-1')

fsp = np.array(data["FSP.1"])
ace = np.array(data["ACE.1"])
dbf = np.array(data["DBF.1"])
wnr = np.array(data["WNR.1"])
ufe = np.array(data["UFE.1"])
bpc = np.array(data["BPC.1"])
npa = np.array(data["NPA.1"])


st11 = np.array(data['ST1.1'])
st21 = np.array(data['ST2.1'])
st31 = np.array(data['ST3.1'])
st41 = np.array(data['ST4.1'])
st51 = np.array(data['ST5.1'])

inputMatrix = np.array([fsp,ace,dbf,wnr,bpc,npa]).T

test_X = np.array(np.delete(inputMatrix,np.s_[0:200],0))
train_x = np.array(inputMatrix[0:200])

totalY = st11+st21+st31+st41+st51


train_y = np.array(totalY[0:200])
test_Y = np.array(np.delete(totalY,np.s_[0:200],0))

def calculateRSS(x,y):
    rss = 0
    tss = 0
    for i in range(x):
        rss +=  np.sum(np.square(x[i]-y[i]))
        tss += np.sum(np.square(x[i]-np.mean(y)))
    result = 1- rss/tss
    return result


def calculateFeature4(trainX,trainY,testX,testY):
    forth_rs2 = np.array([])
    for i in range(1,151,1):
        reg3 = RandomForest(n_estimators=i, max_depth=7, max_features=4)
        reg3.fit(trainX, trainY)
        predict = reg3.predict(testX)
        rss = calculateRSS(testY, predict)
        forth_rs2 = np.append(forth_rs2,rss)
    return forth_rs2

def calculateFeatureAuto(trainX,trainY,testX,testY):
    first_rs2 = np.array([])
    for i in range(1,151,1):
        reg1 = RandomForest(n_estimators=i, max_depth=7, max_features="auto")
        reg1.fit(trainX,trainY)
        predict = reg1.predict(testX)
        rss = calculateRSS(testY, predict)
        first_rs2 = np.append(first_rs2, rss)
    return first_rs2

def calculateFeatureSqrt(trainX,trainY,testX,testY):
    second_rs2 = np.array([])
    for i in range(1,151,1):
        reg2 = RandomForest(n_estimators=i, max_depth=7, max_features="sqrt")
        reg2.fit(trainX, trainY)
        predict = reg2.predict(testX)
        rss = calculateRSS(testY, predict)
        second_rs2 = np.append(second_rs2,rss)
    return second_rs2


auto = calculateFeatureAuto(train_x, train_y, test_X, test_Y)
sqrt = calculateFeatureSqrt(train_x, train_y, test_X, test_Y)
forth = calculateFeature4(train_x, train_y, test_X, test_Y)


y_reg1 = RandomForest(n_estimators=150, max_depth=7, max_features=4)
y_reg1.fit(train_x,train_y)
pred = y_reg1.predict(test_X)


y_reg2 = RandomForest(n_estimators=150, max_depth=1, max_features=4)
y_reg2.fit(train_x,train_y)
pred2 = y_reg2.predict(test_X)



plt.ylabel('R^2 Score')
plt.xlabel('Number of Estimators(decision trees)')

arrArange = np.arange(0,150,1)

plt.plot(arrArange, auto, "r", label = 'Auto')
plt.plot(arrArange, sqrt, "b", label = 'Sqrt')
plt.plot(arrArange, forth, "g", label = 'Four')

plt.show()

plt.ylabel('Error of Estimation')
plt.xlabel('Estimation')

y1 = np.array([])
y2 = np.array([])

for i in range(len(test_Y)):
    y1 = np.append(y1, pred-test_Y[i])
    y2 = np.append(y2,pred2-test_Y[i])
    
plt.scatter(pred, y1, "r")
plt.scatter(pred2,y2, "b")
plt.axhline(0, c="black", lw=2)
plt.show()
