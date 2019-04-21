#X = n x p
import numpy as np
from scipy import stats

X1 = np.array([[15,17,24,14],
[17,15,32,26],
[15,14,29,23],
[13,12,10,16],
[20,17,26,28],
[15,21,26,21],
[15,13,26,22],
[13,5,22,22],
[14,7,30,17],
[17,15,30,27],
[17,17,26,20],
[17,20,28,24],
[15,15,29,24],
[18,19,32,28],
[18,18,31,27],
[15,14,26,21],
[18,17,33,26],
[10,14,19,17],
[18,21,30,29],
[18,21,34,26],
[13,17,30,24],
[16,16,16,16],
[11,15,25,23],
[16,13,26,16],
[16,13,23,21],
[18,18,34,24],
[16,15,28,27],
[15,16,29,24],
[18,19,32,23],
[18,16,33,23],
[17,20,21,21],
[19,19,30,28]])
X2 = np.array([[13,14,12,21],
[14,12,14,26],
[12,19,21,21],
[12,13,10,16],
[11,20,16,16],
[12,9,14,18],
[10,13,18,24],
[10,8,13,23],
[12,20,19,23],
[11,10,11,27],
[12,18,25,25],
[14,18,13,26],
[14,10,25,28],
[13,16,8,14],
[14,8,13,25],
[13,16,23,28],
[16,21,26,26],
[14,17,14,14],
[16,16,15,23],
[13,16,23,24],
[2,6,16,21],
[14,16,22,26],
[17,17,22,28],
[16,13,16,14],
[15,14,20,26],
[12,10,12,9],
[14,17,24,23],
[13,15,18,20],
[11,16,18,28],
[7,7,19,18],
[12,15,7,28],
[6,5,6,13]])
[n1, p] = list(X1.shape)
n2 = X2.shape[0]
alpha = 0.01
df = n1 + n2 - 2
print("n1: ",n1," n2: ",n2," p: ",p," alpha: ",alpha," df: ",df)
x1bar = np.mean(X1,axis=0)
x2bar = np.mean(X2,axis=0)
print("x1bar: ",x1bar," x2bar: ",x2bar)
W1 = n1*np.cov(X1.T,bias=True)
print("W1: ")
print(W1)
W2 = n2*np.cov(X2.T,bias=True)
print("W2: ")
print(W2)
Spl = (W1 + W2)/df
print("Spl: ")
print(Spl)
Tsq = (n1*n2*(x1bar-x2bar).T.dot(np.linalg.inv(Spl).dot(x1bar - x2bar)))/(n1+n2)
Tsqcritical = (df*p*stats.f.ppf(1-alpha,p,df-p+1))/(df - p + 1)
print("Tsp: ",Tsq," Tsqcritical: ",Tsqcritical)
if Tsq > Tsqcritical:
    print("Tsq > Tsqcritical: Reject Hypothesis")
elif Tsq < Tsqcritical:
    print("Tsq < Tsqcritical: Accept Hypothesis")
else:
    print("Tsq = Tsqcritical: Marginal, choose")
