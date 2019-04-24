import numpy as np
import math
from scipy import stats

alpha = 0.05
X = np.array([[1,162,23,3],
[1,162,23,8],
[1,162,30,5],
[1,162,30,8],
[1,172,25,5],
[1,172,25,8],
[1,172,30,5],
[1,172,30,8],
[1,167,27.5,6.5],
[1,177,27.5,6.5],
[1,157,27.5,6.5],
[1,167,22.5,6.5],
[1,167,22.5,6.5],
[1,167,27.5,9.5],
[1,167,27.5,3.5],
[1,177,20,6.5],
[1,177,20,6.5],
[1,160,34,7.5],
[1,160,34,7.5]])


print("X: ")
print(X)

Y = np.array([[41.5, 45.9, 11.2],
[33.8, 53.3, 11.2],
[27.7, 57.5, 12.7],
[21.7, 58.8, 16.0],
[19.9, 60.6, 16.2],
[15.0, 58.6, 22.6],
[12.2, 58.6, 24.5],
[4.3, 52.4, 38.0],
[19.3, 56.9, 21.3],
[6.4, 55.4, 30.8],
[37.6, 46.9, 14.7],
[18.0, 57.3, 22.2],
[26.3, 55.0, 18.3],
[9.9, 58.9, 28.0],
[25.0, 50.3, 22.1],
[14.1, 61.1, 23.0],
[15.2, 62.9, 20.7],
[15.9, 60.0, 22.1],
[19.6, 60.6, 19.3]])

print("")
print("Y: ")
print(Y)

p = Y.shape[1]
n = Y.shape[0]
q = X.shape[1] -1

print("")
print("p: ",p," q: ",q," n: ",n)
print("")

beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
print("Beta: ")
print(beta)
print("")

ybar = np.mean(Y, axis=0)
print("Ybar: ")
print(ybar)
print("")

E = Y.T.dot(Y) - beta.T.dot(X.T.dot(Y))
H = beta.T.dot(X.T.dot(Y)) - n*np.outer(ybar, ybar)
dfE = n-q-1
dfH = q
print("E: df: ",n-q-1)
print(E)
print("")

print("H: df: ",q)
print(H)
print("")

l = np.linalg.det(E)/np.linalg.det(E+H)
print("lambda: ",l)
print("")

t = math.sqrt((p**2 * dfH - 4)/(p**2 + dfH**2 - 5))
w = dfE + dfH - (p + dfH + 1)/2
df1 = p*dfH
df2 = w*t - (p*dfH - 2)/2
F = ((1 - math.pow(l,1/t))*df2)/(math.pow(l,1/t)*df1)
Fcritical = stats.f.ppf(1-alpha, df1, df2)
pvalue = stats.f.cdf(F, df1, df2)

if not F == 0:
    print("t: ",t," w: ",w," df1: ",df1," df2: ", df2,"F: ", F, " Fcritical: ", Fcritical," pvalue: ",pvalue)
    if F > Fcritical:
        print("F > Fcritical: Reject Hypothesis")
    elif F < Fcritical:
        print("F < Fcritical: Accept Hypothesis")
    else:
        print("F = Fcritical: Marginal, choose")

X1 = np.take(X,[0,1,2],axis=1)
print("X1: ")
print(X1)
print("")

beta1 = np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(Y)
print("beta1: ")
print(beta1)

H1 = Y.T.dot(Y) - beta1.T.dot(X1.T.dot(Y))
l1 = np.linalg.det(E)/np.linalg.det(H1)
print("lambda: ",l1)

df1 = p
df2 = n - q  - p
F = ((1 - l1)*df2)/(l1*df1)
Fcritical = stats.f.ppf(1-alpha, df1, df2)
pvalue = stats.f.cdf(F, df1, df2)

if not F == 0:
    print("df1: ",df1," df2: ", df2,"F: ", F, " Fcritical: ", Fcritical," pvalue: ",pvalue)
    if F > Fcritical:
        print("F > Fcritical: Reject Hypothesis")
    elif F < Fcritical:
        print("F < Fcritical: Accept Hypothesis")
    else:
        print("F = Fcritical: Marginal, choose")
