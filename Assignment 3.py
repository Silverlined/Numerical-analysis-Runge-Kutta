# Modelling & Simulation, Semester 4 Assignment 3
# Implementation of Newton-Raphson's method for finding roots of an equation
# Implementation of Simpson's rule for computing definite integrals

import numpy as np
from math import e
from matplotlib import pyplot
from pylab import plot, show, axis

epsilon = 1e-8

H = 30  # Height of the mast

force = lambda z: 200 * (z / (5 + z)) * e ** (-2 * z / H)


def newton_raphson(func, x0, a):
    x = x0
    n = 0
    while abs(func(x)) >= epsilon:
        x = 1 / 2 * (x + a / x)
        n += 1
    print("Found solution: x =", x, "\nAfter", n, "iterations\n")
    return x


def simpson(func, a, b, N):
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    h = (b - a) / N  # stepsize
    time = np.linspace(a, b, N + 1)
    y = func(time)
    return h / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])


def main():
    x0 = 3
    a = 10
    equation = lambda x: x ** 2 - a
    newton_raphson(equation, x0, a)
    print(
        "1/3 Simpson's Rule: The total force F exerted on the mast,\ni.e. the definite integral of the force function evaluated from 0 to H(30), is equal to:",
        simpson(force, 0, H, 50),
    )

    last = 0
    intervals = []
    for i in range(6, 100, 2):
        new = simpson(force, 0, H, i)
        if new - last <= 1e-4:
            intervals.append(i)
        last = new
    if intervals:
        print("\nNumber of intervals achieving a percentage error of <=0.01%:")
        print(intervals[::])

    fig1, axis1 = pyplot.subplots()
    axis1.grid()
    pyplot.xlabel("height (z)")
    pyplot.ylabel("force (f)")

    x = range(0, 10)
    y = []
    for i in x:
        y.insert(i, force(i))

    axis1.plot(x, y, "k-", label="")
    # axis1.legend(loc="upper left")
    axis1.text(
        0,
        70,
        "Step size: 1 (units of time)\n",
        bbox=dict(facecolor="red", alpha=0.5),
    )
    pyplot.show()


if __name__ == "__main__":
    main()
