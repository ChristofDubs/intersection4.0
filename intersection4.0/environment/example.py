import time
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from intersection import Intersection
from car import Car, CarParams
from param import intersection_params, plot_street
from definitions import Target, Action
from collision_checker import CollisionChecker

intersection = Intersection(intersection_params)
collision_checker = CollisionChecker(intersection)
cars = []

plt.figure(0)
start_time = time.time()
alpha = 0
collision_pairs = []
while True:
    plt.figure(0)
    plt.cla()
    plot_street(plt)

    collisions_flat = [car for pair in collision_pairs for car in pair]

    for i, car in enumerate(cars):
        car.set_action(Action(0))
        in_collision = i in collisions_flat
        style = 'g-' if not in_collision else 'r-'
        car.plot_interpolated(plt, intersection, alpha, style)
        # car.plot(plt, intersection, style)
        if in_collision:
            car.node_idx = -1

    plt.show(block=False)
    plt.pause(0.05)

    # delete inactive cars
    cars[:] = [car for car in cars if car.is_active()]

    collision_pairs = collision_checker.check_collisions(cars)

    time_passed = time.time() - start_time
    start_time = time.time()
    alpha += time_passed
    if alpha > 1:
        for car in cars:
            if car.is_active():
                car.execute_action()
        alpha = np.fmod(alpha, 1)

        for quadrant in range(4):
            x = randint(0, 5)
            if x > 2:
                continue
            target = x % 3

            cars.append(Car(CarParams()))
            cars[-1].spawn(quadrant, Target(target), 4, intersection)
