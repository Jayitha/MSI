#Multiivariate one way MANOVA
import numpy as np
from scipy import stats
import math

alpha = 0.05
X = [np.array([[24.0, 3.5],
[13.3, 3.5],
[12.2, 4.0],
[14.0, 4.0],
[22.2, 3.6],
[16.1, 4.3],
[27.9, 5.2]])]

X.append(np.array([[7.4, 3.5],
[13.2, 3.0],
[8.5, 3.0],
[10.1, 3.0],
[9.3, 2.0],
[8.5, 2.5],
[4.3, 1.5]]))

X.append(np.array([[16.4, 3.2],
[24.0, 2.5],
[53.0, 1.5],
[32.7, 2.6],
[42.8, 2.0]]))

X.append(np.array([[25.1, 2.7],
[5.9, 2.3]]))

print("X: ")
print(X)
g = len(X)
#g = 3
p = X[0].shape[1]
#p = 4
n = np.array([x.shape[0] for x in X])
#n = np.array([271, 138, 107])
print("p: ",p," g: ",g," nl: ",n)
xgbar = np.array([np.mean(x,axis=0) for x in X])
# xgbar = np.array([[2.066, 0.480, 0.082, 0.360],
# [2.167, 0.596, 0.124, 0.418],
# [2.273, 0.521, 0.125, 0.383]])
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
    #print(B)
    for j in range(n[i]):
        W = W + np.outer((X[i][j] - xgbar[i]),(X[i][j] - xgbar[i]))
        S = S + np.outer((X[i][j] - xbar),(X[i][j] - xbar))
# S1 = np.array([[0.291, -0.001, 0.002, 0.010],
# [-0.001, 0.11, 0.000, 0.003],
# [0.002, 0.000, 0.001, 0.000],
# [0.010, 0.003, 0.000, 0.010]])
#
# S2 = np.array([[0.561, 0.011, 0.001, 0.037],
# [0.011, 0.025, 0.004, 0.007],
# [0.001, 0.004, 0.005, 0.002],
# [0.037, 0.007, 0.002, 0.019]])
#
# S3 = np.array([[0.261, 0.030, 0.003, 0.018],
# [0.030, 0.017, -0.000, 0.006],
# [0.003, -0.000, 0.004, 0.001],
# [0.018, 0.006, 0.001, 0.013]])
#
# if np.array_equal(S1, S1.T):
#     print("S1, okay")
# else:
#     print("S1, not okay")
#     print(S1 - S1.T)
# if np.array_equal(S2, S2.T):
#     print("S2, okay")
# else:
#     print("S2, not okay")
#     print(S2 - S2.T)
# if np.array_equal(S3, S3.T):
#     print("S3, okay")
# else:
#     print("S3, not okay")
# W = (n[0] - 1)*S1 + (n[1] - 1)*S2 + (n[2]-1)*S3
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
    print(S - (B+W))
    print("Could be floating point error")

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
    print("here")
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

print("Chisq: ",chisq, " chisqcritical: ", chisqcritical, "pvaluefromchi: ",pvaluefromchi)
if chisq > chisqcritical:
    print("chisq > chisqcritical: Reject Hypothesis")
elif chisq < chisqcritical:
    print("chisq < chisqcritical: Accept Hypothesis")
else:
    print("chisq = chisqcritical: Marginal, choose")

#Confidence Intervals when hypothesis is rejected
Tou = xgbar - xbar
print("Tou: ")
print(Tou)
for k in range(1,g):
    for l in range(k):
        for i in range(p):
            CI = abs(stats.t.ppf((alpha/(p*g*(g-1))),np.sum(n)-g)*math.sqrt((W[i,i]/(np.sum(n)-g))*(1/n[k]+1/n[l])))
            print("CI for Tou(",k,i,") - Tou(",l,i,") = (",xgbar[k,i] - xgbar[l,i] - CI," , ",xgbar[k,i] - xgbar[l,i] + CI,")")

S1 = np.zeros((p,p))
for j in range(g):
    for i in range(n[j]):
        S1 = S1 + np.outer(X[j][i],X[j][i])

print(S1/(np.sum(n)-g))
