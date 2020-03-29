import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data
data = pd.read_csv("team_1.csv", encoding = "ISO-8859-1")
test = pd.read_csv("team_2.csv", encoding = "ISO-8859-1")
#takes required field
x_d = np.array(data["Age"])
y_d= np.array(data["Experience"])
x_t = np.array(test["Age"])
y_t = np.array(test["Experience"])
rss_score = 0
rss_score1 = 0
tss_score = 0
tss_score1 = 0
def coef(x,y):
    totalX = 0
    totalY = 0
    for i in range(len(x)):
        totalX += x[i] 
        totalY += y[i]
        
    mean_X = totalX/len(x)
    mean_Y = totalY/len(y)
    cov1 = 0
    cov2 = 0
    # calculating cross-deviation and deviation about x
    for i in range (0, len(x)):
        cov1 += ((x[i]-mean_X)*(y[i]-mean_Y))
    
    for i in range(0, len(x)):
        cov2 += (x[i]-mean_X)**2
    # calculating regression coefficients  
    cov = cov1/cov2
    b1 = cov
    
    b0 = (mean_Y)-((mean_X)*b1)
    
    return [b0, b1]
  
def plot(x,y,c):
    X = np.linspace(np.min(x), np.max(x))

    Y = c[0] + (c[1] * X)
    # plotting the regression line 
    plt.plot(X,Y,color="red")
    # plotting the actual points as scatter plot
    plt.scatter(x,y)
    plt.show()

def rss(x,y,c):
    rss_data = 0
    
    for r in range(len(y)):
        # predicted response vector 
        y_pred = c[0] + (c[1] * x[r])
        rss_data += (y[r] - y_pred)**2  
    return rss_data
def tss(x,y,c):
    tss_data = 0
    totalY = 0
    for i in range(len(x)):
        totalY += y[i]
    y_average = totalY/len(y)
    for r in range(len(y)):
       
        tss_data += (y[r]-y_average)**2
    return tss_data

def rSquare(rss_value,tss_value):
    r_square = 1- rss_value/tss_value
    print(r_square)
    
c = coef(x_d,y_d)
ct = coef(x_t,y_t)

#showing values
plot(x_d,y_d,c)
plot(x_t,y_t,ct)


rss_score = rss(x_t,y_t,c)
print(" Rss Score 1 :" + str(rss_score))
rss_score1 = rss(x_d,y_d,ct) 
print(" Rss Score 2 :" + str(rss_score1))
tss_score = tss(x_t,y_t,c)
print(" Tss Score 1 :" + str(tss_score))
tss_score1 = tss(x_d,y_d,ct) 
print(" Tss Score 2 :" + str(tss_score1))

rSquare(rss_score1,tss_score1)
rSquare(rss_score,tss_score)


    