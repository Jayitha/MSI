from scipy import stats
import numpy as np
import math

#First Index = Factot 1 level
#Second Index = Factor 2 level
#Third Index = observation number
#Fourth Index = p

alpha = 0.05
X = np.array([[[[7.80, 90.4],
[7.10, 88.9],
[7.89, 85.9],
[7.82, 88.8]],
[[9.00, 82.50],
[8.43, 92.40],
[7.65, 82.40],
[7.70, 87.40]],
[[7.28, 79.60],
[8.96, 95.10],
[7.75, 90.20],
[7.80, 88.0]],
[[7.60, 94.1],
[7.00, 86.6],
[7.82, 85.9],
[7.80, 88.8]]],
[[[7.12, 85.1],
[7.06, 89.0],
[7.45, 75.9],
[7.45, 77.9]],
[[8.19, 66.0],
[8.25, 74.5],
[7.45, 83.1],
[7.45, 86.4]],
[[7.15, 81.2],
[7.15, 72.0],
[7.70, 79.9],
[7.45, 71.9]],
[[7.06, 81.2],
[7.04, 79.9],
[7.52, 86.4],
[7.70, 76.4]]]])

print("X: ",X)
[g, b, n, p] = list(X.shape)
print("")
print("g: ",g," b: ",b," n: ",n," p: ",p)
print("")
xgbbar = np.mean(X, axis=2)
xgbar = np.mean(X, axis=(1,2))
xbbar = np.mean(X, axis=(0,2))
xbar = np.mean(X, axis=(0,1,2))
print("xbar: ",xbar)
print("xgbar: ")
print(xgbar)
print("xbbar: ")
print(xbbar)
print("xgbbar: ")
print(xgbbar)
print("")
#Calculating Sum of Squares
#Factor 1
dffact1 = g - 1
dffact2 = b -1
dfint = (g-1)*(b-1)
dfres = g*b*(n-1)
dfcor = g*b*n - 1

SSPfact1 = np.zeros((p,p))
SSPfact2 = np.zeros((p,p))
SSPint = np.zeros((p,p))
SSPres = np.zeros((p,p))
SSPcor = np.zeros((p,p))

for l in range(g):
    SSPfact1 = SSPfact1 + b*n*np.outer((xgbar[l,:] - xbar),(xgbar[l,:] - xbar))
    for k in range(b):
        SSPint = SSPint + n*np.outer((xgbbar[l,k,:] - xgbar[l,:] - xbbar[k,:] + xbar),(xgbbar[l,k,:] - xgbar[l,:] - xbbar[k,:] + xbar))
        for r in range(n):
            SSPres = SSPres + np.outer((X[l,k,r,:] - xgbbar[l,k,:]),(X[l,k,r,:] - xgbbar[l,k,:]))
            SSPcor = SSPcor + np.outer((X[l,k,r,:] - xbar),(X[l,k,r,:] - xbar))

for k in range(b):
    SSPfact2 = SSPfact2 + g*n*np.outer((xbbar[k,:] - xbar),(xbbar[k,:] - xbar))

print("SSPfact1:  df: ",dffact1)
print(SSPfact1)
print("")
print("SSPfact2:  df: ",dffact2)
print(SSPfact2)
print("")
print("SSPint:  df: ",dfint)
print(SSPint)
print("")
print("SSPres:  df: ",dfres)
print(SSPres)
print("")
print("SSPcor:  df: ",dfcor)
print(SSPcor)
print("")
if np.array_equal(SSPcor, SSPfact1 + SSPfact2 + SSPint + SSPres):
    print("SSPcor = SSPfact1 + SSPfact2 + SSPint + SSPres")
else:
    print("SSPcor != SSPfact1 + SSPfact2 + SSPint + SSPres")
    print(SSPcor - (SSPfact1 + SSPfact2 + SSPint + SSPres))
    print("Might be rounding off error, is okay, bruh.")
print("")

print("#############################################################################################")
print("")
print("Performing Interaction Test: If interaction exists, individual factor effect results are ambiguous")

l = (np.linalg.det(SSPres))/(np.linalg.det(SSPint + SSPres))
print("lambda: ",l)
print("")

# print("Chi sq Test: ")
# chisq = -(g*b*(n-1) - ((p + 1 - (g-1)*(b-1))/2)) * math.log(l)
# chisqcritical = stats.chi2.ppf(1-alpha, (g-1)*(b-1)*p)
# pvaluefromchi = stats.chi2.sf(chisq, (g-1)*(b-1)*p)
#
# print("Chisq: ",chisq," chisqcritical: ",chisqcritical," pvalue: ",pvaluefromchi)
# if chisq > chisqcritical:
#     print("chisq > chisqcritical: Hypothesis Rejected")
# elif chisq < chisqcritical:
#     print("chisq < chisqcritical: Hypothesis NOT Rejected")
# else:
#     print("chisq = chisqcritical: Marginal")

df1 = 2*(g - 1)*(b - 1)
df2 = 2*(g*b*(n-1) - 1)
F = (df2 * (1 - math.sqrt(l)))/(df1 * math.sqrt(l))
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

print("#############################################################################################")
print("")
print("Performing Test for significance of factor 1")

l = (np.linalg.det(SSPres))/(np.linalg.det(SSPfact1 + SSPres))
print("lambda: ",l)
print("")

df1 = p
df2 = g*b*(n-1) + (g-1)-p
F = ((1 - l)*df2)/(l*df1)
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

print("#############################################################################################")
print("")
print("Performing for Fact2")

l = (np.linalg.det(SSPres))/(np.linalg.det(SSPfact2 + SSPres))
print("lambda: ",l)
print("")

df1 = 2*(b - 1)
df2 = 2*(g*b*(n-1) - 1)
F = (df2 * (1 - math.sqrt(l)))/(df1 * math.sqrt(l))
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
