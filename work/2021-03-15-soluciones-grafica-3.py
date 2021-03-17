import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


data = pd.read_csv('work/datasets/admission_predict.csv', header=0)

X = pd.DataFrame({
    "x1": data["GRE Score"],
    "x2": data["TOEFL Score"],
    "x3": data["University Rating"],
    "x4": data["SOP"],
    "x5": data["LOR "],
    "x6": data["CGPA"],
    "x7": data["Research"],
})

y = data["Chance of Admit "]

reg = LinearRegression()

reg.fit(X, y)

# Drop

X = X.drop(columns=["x1", "x4"])
reg.fit(X, y)

# 2D

m1 = reg.coef_[3]
m2 = reg.coef_[4]

x1 = X["x6"]
x2 = X["x7"]

b = reg.intercept_

y_predict_1 = [m1 * x1[i] + b for i in range(len(y))]
y_predict_2 = [m2 * x2[i] + b for i in range(len(y))]

# 3D

N = len(y)

X = np.linspace(0, 10, N)
Y = np.linspace(0, 1, N)

X, Y = np.meshgrid(X, Y)

Z = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        Z[i][j] = m1 * X[i][j] + m2 * Y[i][j] + b

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.scatter(x1, x2, y, c="magenta")
plt.show()

