import numpy as np
from math import sqrt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from matplotlib import pyplot
from pylab import plot, show, axis

p = 4e-2  # probability of transmission
c = 2.5e-7  # per capita contact rate with susceptible individual
beta = p * c  # transmission rate const
alpha = 2e-2  # recovery rate const
step_size = 1
total_population = 17e6


def thousands(x, pos):
    "The two args are the value and tick position"
    if x >= 1e6:
        return "%1.1fM" % (x / 1e6)
    else:
        return "%dK" % (x / 1000)


formatter = FuncFormatter(thousands)

# Implementation of 4th-order Runge Kutta method, returns a dictionary
# with time points of evaluation as "keys" and function output at those time points as "values":
def runge_kutta4SIR(funcS, funcI, funcR, s0, i0, r0, t0, t_final, h):
    solutionSusceptible = {}
    solutionInfected = {}
    solutionRecovered = {}
    t, s, i, r = t0, s0, i0, r0
    while t <= t_final:
        solutionSusceptible[t] = s
        solutionInfected[t] = i
        solutionRecovered[t] = r
        print(
            "day",
            t,
            "\tTotal cases:",
            round(solutionInfected[t]),
            "\tRecovered+Deaths:",
            round(solutionRecovered[t]),
        )
        k1 = funcS(s, i)
        l1 = funcI(s, i)
        m1 = funcR(i)
        k2 = funcS(s + 0.5 * k1, i + 0.5 * l1)
        l2 = funcI(s + 0.5 * k1, i + 0.5 * l1)
        m2 = funcR(i + 0.5 * l1)
        k3 = funcS(s + 0.5 * k2, i + 0.5 * l2)
        l3 = funcI(s + 0.5 * k2, i + 0.5 * l2)
        m3 = funcR(i + 0.5 * l2)
        k4 = funcS(s + k3, i + l3)
        l4 = funcI(s + k3, i + l3)
        m4 = funcR(i + l3)
        s += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        i += h * (l1 + 2 * l2 + 2 * l3 + l4) / 6
        r += h * (m1 + 2 * m2 + 2 * m3 + m4) / 6
        t += h
    return solutionSusceptible, solutionInfected, solutionRecovered


def getNewSusceptible(susceptibleNow, infectedNow):
    return -(beta * susceptibleNow * infectedNow)


def getNewInfected(susceptibleNow, infectedNow):
    return min((beta * susceptibleNow - alpha) * infectedNow, total_population)


def getRemoved(infectedNow):
    return alpha * infectedNow


def main():
    fig1, axis1 = pyplot.subplots()
    axis1.grid()
    pyplot.xlabel("Time (days)")
    pyplot.ylabel("Number of active cases")
    axis1.yaxis.set_major_formatter(formatter)
    # The loop can be used to plot multiple solutions based on different incrementing step-sizes.
    dataSusceptible, dataInfected, dataRecovered = runge_kutta4SIR(
        getNewSusceptible,
        getNewInfected,
        getRemoved,
        total_population,
        607,
        0,
        0,
        60,
        step_size,
    )
    data_list = sorted(dataSusceptible.items())
    data_list2 = sorted(dataInfected.items())
    data_list3 = sorted(dataRecovered.items())
    x, y = zip(*data_list)
    x2, y2 = zip(*data_list2)
    x3, y3 = zip(*data_list3)

    # axis1.plot(x, y, "o-", label="Susceptible")
    axis1.plot(x2, y2, "k-", label="Infectious")
    axis1.plot(x3, y3, "r-", label="Recovered+Dead")
    axis1.legend(loc="upper left")
    axis1.text(
        0,
        1e6,
        "Step size: 1 day\nR0=8\nTransmission rate(beta)=1e-8\nRecovery rate(alpha)=2e-2",
        bbox=dict(facecolor="red", alpha=0.4),
    )
    pyplot.show()


if __name__ == "__main__":
    main()
