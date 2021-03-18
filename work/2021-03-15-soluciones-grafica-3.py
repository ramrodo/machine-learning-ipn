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

x1 = np.array(data["CGPA"])
x2 = np.array(data["Research"])
N = len(y)

x1_m, x2_m = np.meshgrid(
    np.linspace(0, 10, N),
    np.linspace(0, 1, N),
)

X_m = np.array([
    x1_m.ravel(),
    x2_m.ravel()
]).T

reg = LinearRegression()
reg.fit(X, y)

print("coef (m1, m2): ", reg.coef_)
print("intercept (b): ", reg.intercept_)
print("R^2 (score): ", reg.score(X, y))

#y_predict = reg.coef_[0] * x1_m + reg.coef_[1] * x2_m + reg.intercept_
y_predict = reg.predict(X_m)
#print("y_predict: ", y_predict)

#plt.scatter(x1_m, x2_m, c=y_predict)
#plt.scatter(x1, x2, c=y)

# 3D

fig = plt.figure()

ax = fig.add_subplot(111, projection="3d")

ax.scatter(x1, x2, y)
ax.scatter(x1_m, x2_m, y_predict)
