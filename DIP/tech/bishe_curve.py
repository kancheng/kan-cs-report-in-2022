import numpy as np
import matplotlib.pyplot as plt

k = 39
s, e = 49, 1


def get_list(k):
    x = list(range(1, k+2))
    y = [e - (s - e) * np.log(((ix - 1) * (np.exp(1) - 1) / (k - 1) + 1) / np.exp(1)) for ix in x]
    return x, y


def get_rho_sigma(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=1.0):
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    rhos = list(map(lambda x: 0.23*(sigma**2)/(x**2), sigmas))
    return rhos, sigmas


r1, y1 = get_rho_sigma(iter_num=40, modelSigma1=49, modelSigma2=1)
x1 = list(range(1, 40+1))

r2, y2 = get_rho_sigma(iter_num=24, modelSigma1=49, modelSigma2=1)
x2 = list(range(1, 24+1))

r3, y3 = get_rho_sigma(iter_num=8, modelSigma1=49, modelSigma2=1)
x3 = list(range(1, 8+1))

plt.plot(x1, r1, color="deeppink", linewidth=2, linestyle=':', label='iterations=40', marker='o')
plt.plot(x2, r2, "ob", color="darkblue", linewidth=1, linestyle='--', label='iterations=24', marker='+')
plt.plot(x3, r3, "*g", color="goldenrod", linewidth=1.5, linestyle='-', label='iterations=8', marker='*')
plt.ylabel("alpha")

# plt.plot(x1, r1)
plt.xlabel("iterations")
plt.legend()
plt.show()

