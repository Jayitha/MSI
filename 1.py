#X = n x p
import numpy as np
from scipy import stats
X = np.array([[69, 153],
[74, 175],
[68, 155],
[70, 135],
[72, 172],
[67, 150],
[66, 115],
[70, 137],
[76, 200],
[68, 130],
[72, 140],
[79, 265],
[74, 185],
[67, 112],
[66, 140],
[71, 150],
[74, 165],
[75, 185],
[75, 210],
[76, 220]])
mu0 = np.array([70, 170])
sigma = np.array([[20, 100],[100, 1000]])
alpha = 0.05
[n, p] = list(X.shape)
df = p
xbar = np.mean(X, axis=0)
print("xbar: ")
print(xbar)
zsq = n*np.transpose(xbar - mu0).dot(np.linalg.inv(sigma).dot(xbar - mu0))
print("zsq: ",zsq)
chisq = stats.chi2.ppf(1-alpha, df)
print("chisq: ",chisq)
if zsq > chisq:
    print("zqs > chisq: Reject Hypothesis")
elif zsq < chisq:
    print("zsq < chisq: Accept Hypothesis")
else:
    print("zsq = chisq: Marginal, choose")
