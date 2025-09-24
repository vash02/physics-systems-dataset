# Markus Buchholz, 2023 (ported to Python)
# Aizawa attractor with RK4 integration

from typing import Tuple, List
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# --------------------------------------------------------------------------------
# Parameters (match the C++ values)
a = 0.95
b = 0.7
c = 0.6
d = 3.5
e = 0.25
f = 0.1
dt = 0.01

# --------------------------------------------------------------------------------
# dot x
def function1(x: float, y: float, z: float) -> float:
    return (z - b) * x - d * y

# dot y
def function2(x: float, y: float, z: float) -> float:
    return d * x + (z - b) * y

# dot z
def function3(x: float, y: float, z: float) -> float:
    return c + a * z - (z ** 3) / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * (x ** 3)

def method_runge_kutta_1diff(
    steps: int = 10000
) -> Tuple[List[float], List[float], List[float], List[float]]:
    diffEq1: List[float] = []
    diffEq2: List[float] = []
    diffEq3: List[float] = []
    time: List[float] = []

    # init values (match C++)
    x1 = 0.1  # x
    x2 = 0.0  # y
    x3 = 0.0  # z
    t = 0.0

    diffEq1.append(x1)
    diffEq2.append(x2)
    diffEq3.append(x3)
    time.append(t)

    for _ in range(steps):
        t = t + dt

        k11 = function1(x1, x2, x3)
        k12 = function2(x1, x2, x3)
        k13 = function3(x1, x2, x3)

        k21 = function1(x1 + dt * 0.5 * k11, x2 + dt * 0.5 * k12, x3 + dt * 0.5 * k13)
        k22 = function2(x1 + dt * 0.5 * k11, x2 + dt * 0.5 * k12, x3 + dt * 0.5 * k13)
        k23 = function3(x1 + dt * 0.5 * k11, x2 + dt * 0.5 * k12, x3 + dt * 0.5 * k13)

        k31 = function1(x1 + dt * 0.5 * k21, x2 + dt * 0.5 * k22, x3 + dt * 0.5 * k23)
        k32 = function2(x1 + dt * 0.5 * k21, x2 + dt * 0.5 * k22, x3 + dt * 0.5 * k23)
        k33 = function3(x1 + dt * 0.5 * k21, x2 + dt * 0.5 * k22, x3 + dt * 0.5 * k23)

        k41 = function1(x1 + dt * k31, x2 + dt * k32, x3 + dt * k33)
        k42 = function2(x1 + dt * k31, x2 + dt * k32, x3 + dt * k33)
        k43 = function3(x1 + dt * k31, x2 + dt * k32, x3 + dt * k33)

        x1 = x1 + dt / 6.0 * (k11 + 2.0 * k21 + 2.0 * k31 + k41)
        x2 = x2 + dt / 6.0 * (k12 + 2.0 * k22 + 2.0 * k32 + k42)
        x3 = x3 + dt / 6.0 * (k13 + 2.0 * k23 + 2.0 * k33 + k43)

        diffEq1.append(x1)
        diffEq2.append(x2)
        diffEq3.append(x3)
        time.append(t)

    return diffEq1, diffEq2, diffEq3, time

# --------------------------------------------------------------------------------
def plot2D(xX: List[float], yY: List[float]) -> None:
    plt.figure()
    plt.title("Aizawa attractor")
    plt.plot(xX, yY, label="solution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

def plot3D(xX: List[float], yY: List[float], zZ: List[float]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xX, yY, zZ)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

# --------------------------------------------------------------------------------
if __name__ == "__main__":
    xX, yY, zZ, t = method_runge_kutta_1diff()
    # Toggle which plot you want:
    # plot3D(xX, yY, zZ)
    plot2D(xX, yY)
