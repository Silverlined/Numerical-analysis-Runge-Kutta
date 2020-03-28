# Modelling & Simulation, Semester 4 Assignment 2 - Model and simulated a spring-dashpot system
# Implementation of Runge Kutta method for second-order differenctial equations
import numpy as np
from math import sqrt
from matplotlib import pyplot
from pylab import plot, show, axis

k = 1  # N/m
b = 0.5  # kg/s
Fmax = 3  # N
m = 3  # kg

t0 = 0
t_final = 180
h = 1  # step size
num = (t_final - t0) / 1
time = np.linspace(t0, t_final, num)
input = Fmax * np.sin(time)

# Implementation of 4th-order Runge Kutta method, returns a dictionary
# with time points of evaluation as "keys" and calculated water level as "values":
def runge_kutta(func, y0, y0_prime, h):
    solution = {}
    t, y, y_prime = t0, y0, y0_prime
    n = 0
    while t < t_final - h:
        solution[t] = func(input[n], y, y_prime)
        k1 = 1 / 2 * h ** 2 * func(input[n], y, y_prime)
        k2 = (
            1
            / 2
            * h ** 2
            * func(
                (input[n] + input[n + 1]) / 2,
                y + 1 / 2 * h * y_prime + 1 / 4 * k1,
                y_prime + k1 / h,
            )
        )
        k3 = (
            1
            / 2
            * h ** 2
            * func(
                (input[n] + input[n + 1]) / 2,
                y + 1 / 2 * h * y_prime + 1 / 4 * k1,
                y_prime + k2 / h,
            )
        )
        k4 = (
            1
            / 2
            * h ** 2
            * func(input[n + 1], y + h * y_prime + k3, y_prime + 2 * k3 / h)
        )
        P = 1 / 3 * (k1 + k2 + k3)
        Q = 1 / 3 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        y += h * y_prime + P
        y_prime += Q / h
        n += 1
    return solution


# Model of the water tank system:
def acceleration(force, displacement, velocity):
    return 1 / m * (force - b * velocity - k * displacement)


def main():
    fig1, axis1 = pyplot.subplots()
    axis1.grid()
    pyplot.xlabel("Time")
    pyplot.ylabel("Water level (m)")

    data = runge_kutta(acceleration, 0, 0, 1)
    data_list = sorted(data.items())
    x, y = zip(*data_list)

    axis1.plot(x, y, "k-", label="Runge Kutta's Method 2nd Order DE")
    axis1.legend(loc="upper left")
    axis1.text(
        30,
        45,
        "Step Size: 1 (units of time)",
        bbox=dict(facecolor="red", alpha=0.5),
    )
    pyplot.show()


if __name__ == "__main__":
    main()
