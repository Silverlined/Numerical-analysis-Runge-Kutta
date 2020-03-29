# Modelling & Simulation, Semester 4 Assignment 2 - Model and simulated a spring-dashpot system
# Implementation of Runge Kutta method for second-order differenctial equations
import numpy as np
from math import sqrt
from matplotlib import pyplot
from pylab import plot, show, axis

k = 1  # spring constant, N/m
b = 0.5  # damping constant, kg/s
Fmax = 3  # amplitude of the force, N
m = 3  # mass of the object, kg

t0 = 0  # starting point of time
t_final = 180  # final point of time
h = 1  # step size
num = (t_final - t0) / h  # number of samples in the given time frame
time = np.linspace(t0, t_final, num)
input = Fmax * np.sin(time)  # function of the applied sin force
input2 = Fmax  # constant force


def getInputForce(n):
    return input[n] if n <= 63 else 0


# Implementation of a Runge Kutta method for 2nd Order DE, returns a dictionary
def runge_kutta(func, y0, y0_prime, h):
    solution = {}
    t, y, y_prime = t0, y0, y0_prime
    n = 0
    while t < t_final - h:
        solution[t] = y
        k1 = 1 / 2 * h ** 2 * func(getInputForce(n), y, y_prime)
        k2 = (
            1
            / 2
            * h ** 2
            * func(
                (getInputForce(n) + getInputForce(n + 1)) / 2,
                y + 1 / 2 * h * y_prime + 1 / 4 * k1,
                y_prime + k1 / h,
            )
        )
        k3 = (
            1
            / 2
            * h ** 2
            * func(
                (getInputForce(n) + getInputForce(n + 1)) / 2,
                y + 1 / 2 * h * y_prime + 1 / 4 * k1,
                y_prime + k2 / h,
            )
        )
        k4 = (
            1
            / 2
            * h ** 2
            * func(
                getInputForce(n + 1), y + h * y_prime + k3, y_prime + 2 * k3 / h
            )
        )
        P = 1 / 3 * (k1 + k2 + k3)
        Q = 1 / 3 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        y += h * y_prime + P
        y_prime += Q / h
        n += 1
    return solution


# Model of the spring-dashpot system:
def acceleration(force, displacement, velocity):
    return 1 / m * (force - b * velocity - k * displacement)


def main():
    fig1, axis1 = pyplot.subplots()
    axis1.grid()
    pyplot.xlabel("Time")
    pyplot.ylabel("Displacement")

    data = runge_kutta(acceleration, 0, 0, 1)
    data_list = sorted(data.items())
    x, y = zip(*data_list)

    axis1.plot(x, y, "k-", label="Runge Kutta's Method 2nd Order DE")
    axis1.legend(loc="upper left")
    axis1.text(
        10,
        2,
        "Step Size: 1 (units of time)\nSystem response to a sinusoidal input",
        bbox=dict(facecolor="red", alpha=0.5),
    )
    pyplot.show()


if __name__ == "__main__":
    main()
