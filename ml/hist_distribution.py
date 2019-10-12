from scipy.stats import gamma
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
#from statsmodels.distributions.empirical_distribution import ECDF
from scipy.integrate import quad
from scipy.optimize import fsolve

a = 10 # shape
b = 2 # scale
n = 1000 # size

# task 1
samples_numpy = np.random.gamma(a, b, n)
samples_scipy = gamma.rvs(a=a, scale=b, size=n)

# task 2

print(gamma.ppf(0.01, a))
print(gamma.ppf(0.99, a))
x = np.linspace(gamma.ppf(0.000001, a), gamma.ppf(0.99999999999, a), n)
plt.plot(x, gamma.pdf(x, a, loc=0, scale=b))
plt.hist(samples_scipy, normed=True)
plt.show()

loc = 30
scale = 3

samples_numpy = np.random.normal(loc, scale, n)
samples_scipy = norm.rvs(loc=loc, scale=scale, size=n)

print(norm.ppf(0.01, loc=loc, scale=scale))
print(norm.ppf(0.99, loc=loc, scale=scale))
x = np.linspace(norm.ppf(0.00000001, loc=loc, scale=scale), norm.ppf(0.9999, loc=loc, scale=scale), n)
plt.plot(x, norm.pdf(x, loc=loc, scale=scale))
plt.hist(samples_scipy, normed=True)
plt.show()

x = [0, 1, 4, 7, 10, 12, 13, 16, 17, 19, 20, 23, 30, 33, 35, 40, 43, 46, 48]
y= [0, 0.001, 0.002, 0.004, 0.1, 0.1, 0.2, 0.15, 0.16, 0.24, 0.27, 0.3, 0.29, 0.31, 0.4, 0.5, 0.48, 0.51, 0.55]
p = np.polyfit(x, y, 2)
p_poly = np.poly1d(p)
x_p = np.linspace(0, 100, 1000)
plt.plot(x, y)
plt.hlines(1, xmin=0, xmax=100, colors='red')
plt.plot(x_p, p_poly(x_p))
plt.show()
# task 3
#fig2, ax2 = plt.subplots()
#cdf = ECDF(samples_scipy)
#ax2.plot(cdf.x, cdf.y, label="statmodels", marker="<", markerfacecolor='none')
#ax2.plot(x, gamma.cdf(x, a, loc=0, scale=b))

# task 4, hack
#def func(x):
#    return gamma.cdf(x, a, loc=0, scale=b) - 0.95

#sol = fsolve(func, 1.0)
#print(sol)

# task 4, non-hack
#def integrand(t):
#    return gamma.pdf(t, a, loc=0, scale=b)

#def func(x):
#    return quad(integrand, 0, x)[0] - 0.95

#sol = fsolve(func, 1.0)
#print(sol)



