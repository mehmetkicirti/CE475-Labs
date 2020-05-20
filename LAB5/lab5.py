import numpy as np
import pandas as pd

data = pd.read_csv('teams_comb.csv', encoding="Latin-1")

experience = np.array(data['Experience'])
age = np.array(data['Age'])
power= np.array(data['Power'])
salary = np.array(data['Salary'])
ones = np.ones(len(power), dtype=int)   
matrix = np.vstack([ones, age, experience, power]).T

       
def calculatePredColumn(input,coefNth):
    result = np.dot(input.T,coefNth)
    return result
    
def r2(matrix, y_pred,d):
    rss = 0
    tss = 0
    n = len(matrix)
    for i in range(len(matrix)):
        rss += np.sum(np.square(matrix[i]-y_pred[i]))
        tss += np.sum(np.square(matrix[i]-np.mean(y_pred)))
    RSquared = 1-((rss/(n-d-1))/(tss/(n-1)))
    return RSquared

def calculatorR2(x,y):
    rss = 0
    tss = 0
    n = len(x)
    #d = 2-1 for d equal to 1
    for i in range(len(x)):
        rss += np.sum(np.square(x[i]-y[i]))
        tss += np.sum(np.square(x[i]-np.mean(y)))
    r2 = 1-(rss/(n-2)/(tss/(n-2)))
    return r2

def coef(x_stack,y_stack):     
    coefficient = np.dot(np.dot(np.linalg.inv(np.dot(x_stack,x_stack.T)),x_stack),y_stack)
    return coefficient

#When Calculated M0 Only
m0 = matrix[:,0]
print("First Adjusted R^2 score: " , r2(salary,m0,0))

resultAdjusted = 0
n=1
while(n<4):
    # take n th matrix 
    m_Nth_Value = matrix[:,n]
    # predColumn value
    predNth= np.dot(m_Nth_Value,np.mean(salary))
    #rSquare Calculated each prediction Column
    rSquareNth = r2(salary,predNth,0)
    #print("M"+str(n)+" RSquare Score",rSquareNth)
    
    # Coef new Matrix with salary
    xNthStack = np.vstack((m0,m_Nth_Value))    
    coefNth= coef(xNthStack,salary)
    #prediction of New Each matrix
    predNthStack = calculatePredColumn(xNthStack,coefNth)
    # adjusted R2 
    adjustedR2 = calculatorR2(salary,predNthStack)
    #print("M"+str(n)+"+M"+str(n)+" New RSquare Score",adjustedR2)
    if(n==3):
        resultAdjusted = adjustedR2
    n+=1

print("Second Adjusted R^2 score: " , resultAdjusted)