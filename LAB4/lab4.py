import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('teams_comb.csv', encoding="ISO-8859-1")

exp_arr = np.array(data['Experience'])
age_arr = np.array(data['Age'])
pow_arr= np.array(data['Power'])
sal_arr = np.array(data['Salary'])
x0 = np.ones(len(age_arr), dtype=int)

X = np.array([x0, age_arr, exp_arr, pow_arr]).T


def coef(trainX,trainY):     
    a1 = trainX.T
    a2 = np.dot(a1, a1.T)
    a3 = np.linalg.inv(a2)
    a4 = np.dot(a3, a1)
    coefficients = np.dot(a4, trainY)
    return coefficients

def predict(X,coefficients):
    y_hat = np.dot(X, coefficients)
    return y_hat

predictArr = []
testArr = []
k = 10
for i in range(k):
    test_x = X[i*4:i*4+4]    
    test_y = sal_arr[i*4:i*4+4]
    train_x= np.delete(X, range(i*4, i*4+4), 0)
    train_y = np.delete(sal_arr,range(i*4, i*4+4), 0)
    Z = predict(test_x,coef(train_x,train_y))
    predictArr.append(Z)
    testArr.append(test_y)

ar_predict = np.array(predictArr)
ar_test = np.array(testArr)
#ypredict-yi
with_cv_mse = ar_predict -ar_test
mesqr = with_cv_mse**2
mse_score = np.mean(mesqr)

without_cv_mse=predict(X,coef(X,sal_arr))
seconderr = without_cv_mse-sal_arr
withoutMse_score = np.mean(seconderr**2)

print("MSE with cross-validation : ",mse_score)
print("MSE without cross-validation : ",withoutMse_score)

plt.title("Residual Error Plot")
plt.xlabel("Predictions")
plt.ylabel("Errors for predictions")
plt.scatter(ar_predict,np.abs(with_cv_mse),color="red")
plt.scatter(without_cv_mse,np.abs(seconderr),color="blue")
plt.legend(["with cv","without cv"])
plt.show()