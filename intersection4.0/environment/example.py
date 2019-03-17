import time
import numpy as np
import matplotlib.pyplot as plt
from intersection import Intersection
from car import Car, CarParams
from param import intersection_params, plot_street
from definitions import SectionIndex, Target, Action


intersection = Intersection(intersection_params)
cars = []

i = 0
j = 0

plt.figure(0)
start_time = time.time()
alpha = 0
while True:
    plt.figure(0)
    plt.cla()
    plot_street(plt)
    for car in cars:
        car.set_action(Action(0))
        car.plot_interpolated(plt, intersection, alpha)

    time_passed = time.time() - start_time
    start_time = time.time()
    alpha += time_passed
    if alpha > 1:
        for car in cars:
            if car.is_active():
                car.execute_action()
        alpha = np.fmod(alpha, 1)

    cars.append(Car(CarParams()))
    cars[-1].spawn(i, Target(j), 6, intersection)

    i = (i + 1) % 4
    j = (j + 1) % 3

    plt.show(block=False)
    plt.pause(0.01)

    if not cars[0].is_active():
        break
