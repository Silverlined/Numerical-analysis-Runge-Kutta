# Modelling & Simulation, Semester 4 Assignment 1 - Model and simulated a water tank system
# Implementation of Runge Kutta methods (incl. Euler's and Ralston's), for solving ODEs

import numpy as np
from math import sqrt
from matplotlib import pyplot
from pylab import plot, show, axis

A = 10  # Cross-sectional area of the tank, 10 square meters
a = 1  # Constant related to the flow rate out of the tank
b = 0.2  # Constant related to the flow rate into the tank
input = (
    150
)  # Constant input related to the flow rate into the tank. (b * input = flow rate in)

# Implementation of Forward Euler's method, returns a dictionary
# with time points of evaluation as "keys" and calculated water level as "values":
def euler(func, y0, t0, t_final, h):
    solution = {}
    t, y = t0, y0
    while t <= t_final:
        solution[t] = y
        t += h
        y += h * func(t, y)
    return solution


# Implementation of Ralston's method (2nd-order method), returns a dictionary
# with time points of evaluation as "keys" and calculated water level as "values":
def runge_kutta2(func, y0, t0, t_final, h):
    solution = {}
    t, y = t0, y0
    while t <= t_final:
        solution[t] = y
        k1 = 1 / 4 * h * func(t, y)
        k2 = 3 / 4 * h * func(t + 2 / 3 * h, y + 2 / 3 * h * func(t, y))
        t += h
        y += k1 + k2
    return solution


# Implementation of 4th-order Runge Kutta method, returns a dictionary
# with time points of evaluation as "keys" and calculated water level as "values":
def runge_kutta4(func, y0, t0, t_final, h):
    solution = {}
    t, y = t0, y0
    while t <= t_final:
        solution[t] = y
        k1 = h * func(t, y)
        k2 = h * func(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * func(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * func(t + h, y + k3)
        t += h
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return solution


# Model of the water tank system:
def changeOfWaterHeight(t, height):
    if height >= 0:
        return 1 / A * (b * input - a * sqrt(2 * 9.8) * sqrt(height))
    else:
        return 0


def main():
    fig1, axis1 = pyplot.subplots()
    axis1.grid()
    pyplot.xlabel("Time")
    pyplot.ylabel("Water level (m)")

    # The loop can be used to plot multiple solutions based on different incrementing step-sizes.
    for step_size in range(4, 5, 1):
        data = euler(changeOfWaterHeight, 40, 0, 180, step_size)
        data2 = runge_kutta2(changeOfWaterHeight, 40, 0, 180, step_size)
        data3 = runge_kutta4(changeOfWaterHeight, 40, 0, 180, step_size)
        data_list = sorted(data.items())
        data_list2 = sorted(data2.items())
        data_list3 = sorted(data3.items())
        x, y = zip(*data_list)
        x2, y2 = zip(*data_list2)
        x3, y3 = zip(*data_list3)

        axis1.plot(x, y, "o-", label="Euler's Method")
        axis1.plot(x2, y2, "k-", label="Ralston's Method")
        axis1.plot(x3, y3, "r-", label="4th Order Runge Kutta")
    axis1.legend(loc="upper left")
    axis1.text(
        30,
        45,
        "Step Size: 4 (units of time)\nA=10\na=1,b=0.2\ninput=150",
        bbox=dict(facecolor="red", alpha=0.5),
    )
    pyplot.show()


if __name__ == "__main__":
    main()
