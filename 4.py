import numpy as np
from scipy import stats
import math

alpha = 0.01
X = [np.array([[9, 3],
[6, 2],
[9, 7]])]
X.append(np.array([[0,4],[2, 0]]))
X.append(np.array([[3,8],[1, 9],[2, 7]]))
print("X: ")
print(X)
g = len(X)
p = X[0].shape[1]
n = np.array([x.shape[0] for x in X])
print("p: ",p," g: ",g," nl: ",n)
xgbar = np.array([np.mean(x,axis=0) for x in X])
print("xgbar: ")
print(xgbar)
xbar = n.T.dot(xgbar)/np.sum(n)
print("xbar: ")
print(xbar)
B = np.zeros((p,p))
W = np.zeros((p,p))
S = np.zeros((p,p))
for i in range(g):
    B = B + n[i]*np.outer((xgbar[i] - xbar),(xgbar[i] - xbar))
    print(B)
    for j in range(n[i]):
        W = W + np.outer((X[i][j] - xgbar[i]),(X[i][j] - xgbar[i]))
        S = S + np.outer((X[i][j] - xbar),(X[i][j] - xbar))
print("B: ")
print(B)
print("W: ")
print(W)
print("S: ")
print(S)

if np.array_equal(S, B+W):
    print("Verified Computation of Variances")
else:
    print("Something\'s wrong")

l = (np.linalg.det(W))/(np.linalg.det(B + W))
print("Lambda: ", l)
F = 0
if p == 1 and g >= 2:
    df1 = g - 1
    df2 = np.sum(n) - g
    F = (df2 * (1 - l))/(df1 * l)
    Fcritical = stats.f.ppf(1-alpha, df1, df2)
    pvalue = stats.f.cdf(F, df1, df2)

elif p == 2 and g >= 2:
    df1 = 2*(g - 1)
    df2 = 2*(np.sum(n) - g - 1)
    F = (df2 * (1 - math.sqrt(l)))/(df1 * math.sqrt(l))
    Fcritical = stats.f.ppf(1-alpha, df1, df2)
    pvalue = stats.f.cdf(F, df1, df2)

elif p >= 1 and g == 2:
    df1 = p
    df2 = np.sum(n) - p - 1
    F = (df2 * (1 - l))/(df1 * l)
    Fcritical = stats.f.ppf(1-alpha, df1, df2)
    pvalue = stats.f.cdf(F, df1, df2)

elif p >= 1 and g == 3:
    df1 = 2*p
    df2 = 2*(np.sum(n) - p - 2)
    F = (df2 * (1 - math.sqrt(l)))/(df1 * math.sqrt(l))
    Fcritical = stats.f.ppf(1-alpha, df1, df2)
    pvalue = stats.f.cdf(F, df1, df2)

else:
    chisq = -(np.sum(n)-1-((p+g)/2))*math.log(l)
    chisqcritical = stats.chi2.ppf(1-alpha, p*(g-1))
    pvaluefromchi = stats.chi2.sf(chisq, p*(g-1))

if not F == 0:
    print("df1: ",df1," df2: ", df2,"F: ", F, " Fcritical: ", Fcritical," pvalue: ",pvalue)
    if F > Fcritical:
        print("F > Fcritical: Reject Hypothesis")
    elif F < Fcritical:
        print("F < Fcritical: Accept Hypothesis")
    else:
        print("F = Fcritical: Marginal, choose")
else:
    print("Chisq: ",chisq, " chisqcritical: ", chisqcritical, "pvaluefromchi: ",pvaluefromchi)
    if chisq > chisqcritical:
        print("chisq > chisqcritical: Reject Hypothesis")
    elif chisq < chisqcritical:
        print("chisq < chisqcritical: Accept Hypothesis")
    else:
        print("chisq = chisqcritical: Marginal, choose")
