import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def h(X):
    a = np.dot(X, w) + b
    return (a >= 0).astype(int)


w = np.array([0, 0.])
b = 0.1


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
z = np.array([0, 0, 0, 1])


# วาดภาพแสดง
plt.gca(aspect=1)
plt.scatter(X[:, 0], X[:, 1], 100, c=z, edgecolor='r', marker='D', cmap='hot')
plt.show()


eta = 0.2  # อัตราการเรียนรู้
print('เริ่มต้น: h(x)=%s, w=%s, b=%s' % (h(X), w, b))
for j in range(100):  # ให้ทำซ้ำสูงสุด 100 ครั้ง
    for i in range(4):
        z_h = z[i] - h(X[i])
        dw = eta*z_h*X[i]
        db = eta*z_h
        w += dw
        b += db
        print('รอบ %d.%d: h(x)=%s, w=%s, b=%s, Δw=%s, Δb=%s' %
              (j+1, i+1, h(X), w, b, dw, db))
    if(np.all(h(X) == z)):  # ถ้าผลลัพธ์ถูกต้องทั้งหมดก็ให้เสร็จสิ้นการเรียนรู้
        break

mx, my = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
mX = np.array([mx.ravel(), my.ravel()]).T
mz = h(mX).reshape(200, -1)
plt.gca(aspect=1, xticks=[0, 1], yticks=[0, 1])
plt.contourf(mx, my, mz, cmap='summer', vmin=0, vmax=1)
plt.scatter(X[:, 0], X[:, 1], 100, c=z, edgecolor='r', marker='D', cmap='hot')
plt.show()


def sigmoid(x):
    return 1/(1+np.exp(-x))


def bandai(x):
    return x > 0


x = np.linspace(-8, 8, 1001)
plt.gca(yticks=np.linspace(0, 1, 11), xlabel='x', ylabel='h')
plt.plot(x, bandai(x), 'r', lw=3)
plt.plot(x, sigmoid(x), 'g', lw=3)
plt.grid(ls=':')
plt.show()


def sigmoid(x):
    return 1/(1+np.exp(-x))


def ha_entropy(z, h):
    return -(z*np.log(h)+(1-z)*np.log(1-h))


# คำตอบของเกต AND
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
z = np.array([0, 0, 0, 1])

w = np.array([0, 0.])  # พารามิเตอร์ตั้งต้น
b = 0
n = len(z)  # จำนวนข้อมูล
eta = 0.8  # อัตราการเรียนรู้
thamsam = 250
entropy = []
for o in range(thamsam):
    for i in range(n):
        ai = np.dot(X[i], w) + b
        hi = sigmoid(ai)
        gai = hi-z[i]
        gwi = gai*X[i]
        gbi = gai
        w -= eta*gwi  # ปรับค่าพารามิเตอร์
        b -= eta*gbi
        J = ha_entropy(z[i], hi)  # คำนวณค่าเสียหายเก็บไว้
        entropy.append(J)

# วาดแสดงการแบ่งพื้นที่
mx, my = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
mX = np.array([mx.ravel(), my.ravel()]).T
mh = np.dot(mX, w) + b
mz = (mh >= 0).astype(int).reshape(200, -1)
plt.gca(aspect=1)
plt.contourf(mx, my, mz, cmap='spring')
plt.scatter(X[:, 0], X[:, 1], 100, c=z, edgecolor='b', marker='D', cmap='gray')
plt.show()


mz = sigmoid(mh).reshape(200, -1)
plt.gca(aspect=1)
plt.contourf(mx, my, mz, 50, cmap='spring')
plt.scatter(X[:, 0], X[:, 1], 100, c=z, edgecolor='b', marker='D', cmap='gray')
plt.show()


plt.plot(entropy, 'r')
plt.xlabel(u'จำนวนรอบ', family='tahoma', size=14)
plt.ylabel(u'ค่าเสียหาย', family='tahoma', size=14)
plt.show()


w = np.array([0, 0.])
b = 0
eta = 0.8
thamsam = 1000
entropy = []
for o in range(thamsam):
    dw = 0
    db = 0
    J = 0
    for i in range(n):
        ai = np.dot(X[i], w) + b
        hi = sigmoid(ai)
        gai = hi-z[i]
        gwi = gai*X[i]
        gbi = gai
        dw -= eta*gwi
        db -= eta*gbi
        J += ha_entropy(z[i], hi)
    w += dw/n
    b += db/n
    entropy.append(J/n)

plt.figure()
plt.plot(entropy, 'r')
plt.xlabel(u'จำนวนรอบ', family='tahoma', size=14)
plt.ylabel(u'ค่าเสียหาย', family='tahoma', size=14)
plt.show()


# Chaper 4 ตัวอย่าง

import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def sigmoid(x):
    return 1/(1+np.exp(-x))


def ha_entropy(z, h):
    return -(z*np.log(h)+(1-z)*np.log(1-h))


class ThotthoiLogistic:
    def __init__(self, eta):
        self.eta = eta

    def rianru(self, X, z, n_thamsam):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.entropy = []
        self.khanaen = []
        for i in range(n_thamsam):
            a = self.ha_a(X)
            h = sigmoid(a)
            J = ha_entropy(z, h)
            ga = (h-z)/len(z)
            self.w -= self.eta*np.dot(ga, X)
            self.b -= self.eta*ga.sum()
            self.entropy.append(J)
            khanaen = ((a >= 0) == z).mean()
            self.khanaen.append(khanaen)

    def thamnai(self, X):
        return (self.ha_a(X) >= 0).astype(int)

    def ha_a(self, X):
        return np.dot(X, self.w) + self.b


d = 25
X1 = np.array([plt.imread(x) for x in glob(
    'assets/images/basic_ch4/0/*.png')]).reshape(-1, 625)
X2 = np.array([plt.imread(x) for x in glob(
    'assets/images/basic_ch4/1/*.png')]).reshape(-1, 625)
# คัดเฉพาะ 900 รูปแรกของแต่ละแบบ นำมารวมกัน
X = np.vstack([X1[:900], X2[:900]])
z = np.arange(2).repeat(900)  # คำตอบ เลข 0 และ 1
tl = ThotthoiLogistic(eta=0.01)  # สร้างออบเจ็กต์ของคลาสการถดถอยโลจิสติก
tl.rianru(X, z, n_thamsam=1000)  # ทำการเรียนรู้
plt.subplot(211, xticks=[])
plt.plot(tl.entropy, 'm')
plt.ylabel(u'เอนโทรปี', family='Tahoma')
plt.subplot(212)
plt.plot(tl.khanaen, 'm')
plt.ylabel(u'คะแนน', family='Tahoma')
plt.xlabel(u'จำนวนรอบ', family='Tahoma')
plt.show()

# นำข้อมูล 100 ตัวที่เหลือมาลองทำนายผล แล้วเทียบกับคำตอบจริง
Xo = np.vstack([X1[900:], X2[900:]])
zo = np.arange(2).repeat(100)
print((tl.thamnai(Xo) == zo).mean())  # ได้ 0.92
