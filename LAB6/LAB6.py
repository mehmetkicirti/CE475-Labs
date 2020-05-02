import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Grand-slams-men-2013.csv')
fsp = dataset['FSP.1']
fsw = dataset['FSW.1']

knotValues1 = [55, 65, 70]
knotValues2 = [60, 75]
knotValues3 = [62]

sortedIndex = fsp.argsort()
sorted_fsp = fsp[sortedIndex]
sorted_fsw = fsw[sortedIndex]

def calculateCoef(X,y_value):
    B = np.linalg.inv(np.dot(X.T, X))
    B = np.dot(B, X.T)
    B = np.dot(B, y_value)
    return B

def calculateX(knotColumns,fsp):
    onesCol = np.ones((1, len(fsp)))
    fsp1 = fsp
    fsp2 = fsp**2
    fsp3 = fsp**3
    X = 0
    if len(knotColumns) == 3:
        X = np.vstack((onesCol, fsp1, fsp2, fsp3, knotColumns[0].T, knotColumns[1].T, knotColumns[2].T)).T
    if len(knotColumns) == 2:
        X = np.vstack((onesCol, fsp1, fsp2, fsp3, knotColumns[0].T, knotColumns[1].T)).T
    if len(knotColumns) == 1:
        X = np.vstack((onesCol, fsp1, fsp2, fsp3, knotColumns[0].T)).T
    return X

def draw(sorted_fsp,sorted_fsw,result1,result2,result3):
    plt.title('Cubic Spline Regression')
    plt.scatter(sorted_fsp, sorted_fsw,s=10)
    plt.plot(sorted_fsp, result1, color = 'green', label = '3 knots')
    plt.plot(sorted_fsp, result2, color = 'red', label = '2 knots')
    plt.plot(sorted_fsp, result3, color = 'blue', label = '1 knot')
    plt.xlabel('First serve percentage of player 1')
    plt.ylabel('First serve won by player 1')

    plt.legend()
    plt.show()

def cubicSplineRegression(x, y, knotValues):
    
    knotColumns = []
    for knot in knotValues:
        res = []
        for x_value in np.nditer(x):
            equation = x_value - knot
            if (equation < 0):
                equation = 0
            res.append(equation)
        
        temp = np.array([res]).transpose()
        col_4 = temp**3
        knotColumns.append(col_4)
        
    X = calculateX(knotColumns,x)
    B = calculateCoef(X, y)
    y_pred = X.dot(B)
    return y_pred   
    
results1 = cubicSplineRegression(sorted_fsp, sorted_fsw, knotValues1)
results2 = cubicSplineRegression(sorted_fsp, sorted_fsw, knotValues2)
results3 = cubicSplineRegression(sorted_fsp, sorted_fsw, knotValues3)
    
draw(sorted_fsp, sorted_fsw, results1, results2, results3)
