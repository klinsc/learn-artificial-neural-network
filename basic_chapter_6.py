from glob import glob
import numpy as np
import matplotlib.pyplot as plt


# def softmax_details(x):
#     x_T = x.T
#     print("x_T=\n", x_T)
#     x_max = x.max(axis=1)
#     print("x_max=\n", x_max)

#     """ Avoiding underflow or overflow errors due to floating point instability """
#     x_T_minus_x_max = x_T - x_max
#     print("x_T_minus_x_max=\n", x_T_minus_x_max)

#     exp_x = np.exp(x_T_minus_x_max)
#     print("exp_x=\n", exp_x)

#     exp_x_sum_0 = exp_x.sum(axis=0)
#     print("exp_x_sum_0=\n", exp_x_sum_0)

#     print("test=\n", exp_x[0, 0]+exp_x[1, 0]+exp_x[2, 0])
#     print("test2=\n", exp_x[0, 1]+exp_x[1, 1]+exp_x[2, 1])
#     print("test3=\n", exp_x[0, 2]+exp_x[1, 2]+exp_x[2, 2])
#     print("test4=\n", exp_x[0, 3]+exp_x[1, 3]+exp_x[2, 3])

#     result = exp_x / exp_x_sum_0
#     print("result=\n", result)
#     result_T = result.T
#     print("result_T=\n", result_T)

#     return result_T


def softmax(x):
    exp_x = np.exp(x.T-x.max(1))
    return (exp_x/exp_x.sum(0)).T


def ha_1h(z, n):
    return (z[:, None] == range(n))


def ha_entropy(z, h):
    return -(np.log(h[z]+1e-10)).mean()


class Prasat:
    def __init__(self, eta):
        self.eta = eta

    def rianru(self, X, z, n_thamsam):
        self.liklum = int(z.max()+1)
        Z = ha_1h(z, self.liklum)
        self.w = np.zeros([X.shape[1], self.liklum])
        self.b = np.zeros(self.liklum)
        self.entropy = []
        self.khanaen = []
        for i in range(n_thamsam):
            a = self.ha_a(X)
            h = softmax(a)
            J = ha_entropy(Z, h)
            ga = (h-Z)/len(z)
            self.w -= self.eta*np.dot(X.T, ga)
            self.b -= self.eta*ga.sum(0)
            self.entropy.append(J)
            khanaen = (h.argmax(1) == z).mean()
            self.khanaen.append(khanaen)

    def thamnai(self, X):
        return self.ha_a(X).argmax(1)

    def ha_a(self, X):
        return np.dot(X, self.w) + self.b


# Example 1
np.random.seed(2)
X = np.random.normal(0, 0.7, [45, 2])
X[:15] += 2
X[30:, 0] += 4

z = np.arange(3).repeat(15)
plt.gca(aspect=1)
plt.scatter(X[:, 0], X[:, 1], 100, c=z, edgecolor='k', cmap='coolwarm')
plt.show()

prasat = Prasat(eta=0.1)
prasat.rianru(X, z, n_thamsam=250)
mx, my = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(
), 200), np.linspace(X[:, 1].min(), X[:, 1].max(), 200))
mX = np.array([mx.ravel(), my.ravel()]).T
mz = prasat.thamnai(mX).reshape(200, -1)
plt.gca(aspect=1, xlim=(X[:, 0].min(), X[:, 0].max()),
        ylim=(X[:, 1].min(), X[:, 1].max()))
plt.contourf(mx, my, mz, cmap='coolwarm', alpha=0.2)
plt.scatter(X[:, 0], X[:, 1], 100, c=z, edgecolor='k', cmap='coolwarm')
plt.show()

plt.subplot(211, xticks=[])
plt.plot(prasat.entropy, 'C9')
plt.ylabel(u'เอนโทรปี', family='Tahoma', size=12)
plt.subplot(212)
plt.plot(prasat.khanaen, 'C9')
plt.ylabel(u'คะแนน', family='Tahoma', size=12)
plt.xlabel(u'จำนวนรอบ', family='Tahoma', size=12)
plt.show()

# Example 2
d = 25
X_ = np.array([plt.imread(x)
              for x in sorted(glob('assets/ruprang-raisi-25x25x1000x5/*/*.png'))])
X = X_.reshape(-1, d*d)
z = np.arange(5).repeat(1000)

prasat = Prasat(eta=0.02)
prasat.rianru(X, z, n_thamsam=1000)
print(prasat.khanaen[-1])

plt.plot(prasat.khanaen, 'C8')
plt.ylabel(u'คะแนน', family='Tahoma', size=12)
plt.xlabel(u'จำนวนรอบ', family='Tahoma', size=12)
plt.show()


prasat.thamnai(X[0:3])