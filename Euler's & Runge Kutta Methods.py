# Implementation of Euler's method and Runge Kutta, for solving ODEs

import numpy as np
from math import sqrt
from matplotlib import pyplot
from pylab import plot, show, axis

A = 10
a = 1
b = 0.2
input = 100

limit = 750
lostFish = 20


def euler(func, y0, t0, t_final, h):
    solution = {}
    t, y = t0, y0
    while t <= t_final:
        solution[t] = y
        t += h
        y += h * func(t, y)
    return solution


def runge_kutta2(func, y0, t0, t_final, h):
    solution = {}
    t, y = t0, y0
    while t <= t_final:
        solution[t] = y
        k1 = h * func(t, y)
        k2 = h * func(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * func(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * func(t + h, y + k3)
        t += h
        y += (k1 + k2 + k3 + k4) / 4
    return solution


def changeOfWaterHeight(t, height):
    if height >= 0:
        return 1 / A * (b * input - a * sqrt(2 * 9.8) * sqrt(height))
    else:
        return 0


def rainbowFishPopulation(t, numberOfFish):
    if numberOfFish >= 0:
        return 0.7 * (1 - numberOfFish / limit) * numberOfFish - lostFish
    else:
        return 0


def main():
    fig1, axis1 = pyplot.subplots()
    # fig2, axis2 = pyplot.subplots()
    axis1.grid()
    # axis2.grid()
    pyplot.xlabel("Time")
    pyplot.ylabel("Signal")
    for step_size in range(2, 3, 1):
        data = euler(changeOfWaterHeight, 40, 0, 20, step_size)
        data2 = runge_kutta2(changeOfWaterHeight, 40, 0, 20, step_size)
        data_list = sorted(data.items())
        data_list2 = sorted(data2.items())
        x, y = zip(*data_list)
        x2, y2 = zip(*data_list2)
        axis1.plot(x, y, "o-", label="Euler's Method")
        axis1.plot(x2, y2, "k-", label="4th Order Runge Kutta")
    axis1.legend(loc="upper left")
    axis1.text(
        10,
        40,
        "Step Size: 2 (units of time)",
        bbox=dict(facecolor="red", alpha=0.5),
    )
    # start, end = axis1.get_xlim()
    # pyplot.xticks(np.arange(start, end, 0.5))
    pyplot.show()


if __name__ == "__main__":
    main()
