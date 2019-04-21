import numpy as np
from scipy import stats
import math
#X1 = np.array([])
#X2 = np.array([])
#n1 = X1.shape[0]
#n2 = X2.shape[0]
n1 = 15
n2 = 20
alpha = 0.05
df = n1 + n2 - 2
#x1bar = int(np.mean(X1))
#x2bar = int(np.mean(X2))
x1bar = 2.514
x2bar = 2.963
print("x1bar: ",x1bar, " x2bar: ",x2bar)
#SS1 = n1*int(np.cov(X1,bias=true))
#SS2 = n2*int(np.cov(X2,bias=true))
#S2pl = (SS1 + SS2)/ df
SS1 = 0
SS2 = 0
S2pl = 0.642**2
print("SS1: ", SS1," SS2: ", SS2," S2pl: ",S2pl)
t = (x1bar - x2bar)/math.sqrt(S2pl*(n1 + n2)/(n1*n2))
print("t: ",t)
pvalue = stats.t.sf(abs(t),df)*2
print("pvalue: ", pvalue)
tcritical = stats.t.ppf(1-alpha/2, df)
print("tcritical: ", tcritical)
if abs(t) > tcritical:
    print("t > tcritical: Reject Hypothesis")
elif abs(t) < tcritical:
    print("t < tcritical: Accept Hypothesis")
else:
    print("t = tcritical: Marginal, choose")
CI = tcritical*(math.sqrt(S2pl*(n1+n2)/(n1*n2)))
print("Margin: ",CI)
print("CI: (",(x1bar - x2bar) - CI," : ",(x1bar - x2bar) + CI,")")
