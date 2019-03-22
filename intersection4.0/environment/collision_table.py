"""Collision table

This module contains a generated table to lookup collisions

author: Christof Dubs
"""
import numpy as np
import matplotlib.pyplot as plt
from param import intersection_params, plot_street
from intersection import Intersection
from car import Car, CarParams
from definitions import Target, Action


# line intersection algorithm:
# https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def ccw(p1, p2, q1):
    return (q1[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (q1[0] - p1[0])


def intersect(p1, p2, q1, q2):
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)


class CollisionTable:
    def __init__(self, intersection):
        self.intersection = intersection
        self.num_cars = 13
        self.cars = [Car(CarParams()) for i in range(self.num_cars)]
        self.start_nodes = [None] * self.num_cars
        self.end_nodes = [None] * self.num_cars
        self.collisions = {}

    def cars_intersect(self, car1, car2):
        if np.linalg.norm(car1.get_pose(self.intersection) - car2.get_pose(self.intersection), ord=2) > (
                0.5 * (car1.param.width + car2.param.width))**2 + (car1.param.back_axis + car2.param.back_axis)**2:
            return False

        car1_outline = car1.get_transformed_outline(car1.get_pose(self.intersection))
        car2_outline = car2.get_transformed_outline(car2.get_pose(self.intersection))

        for i in range(4):
            for j in [0, 2]:
                i_plus = (i + 1) % 4
                if intersect(car1_outline[:, i], car1_outline[:, i_plus],
                             car2_outline[:, j], car2_outline[:, j + 1]):
                    return True

        return False

    def car_arrived(self, car_idx):
        return self.cars[car_idx].route_segment_idx >= 2 and self.cars[car_idx].node_idx > self.end_nodes[car_idx]

    def advance_car(self, car_idx):
        self.cars[car_idx].set_action(Action.KEEP_SPEED)
        self.cars[car_idx].execute_action()

    def calculate_collision_table(self):
        for k in range(3):
            for i, car in enumerate(self.cars):
                idx = i + k
                quadrant = int(idx / 4)
                target = idx % 3
                car.spawn(quadrant, Target(target), 1, intersection)
                self.start_nodes[i] = car.route_segment_lookup[0] - \
                    int(np.ceil(car.param.back_axis / intersection_params.step_size))
                car.node_idx = self.start_nodes[i]
                self.end_nodes[i] = int(
                    np.ceil(
                        (car.param.length -
                         car.param.back_axis) /
                        intersection_params.step_size))

            not_arrived_cars = list(range(1, self.num_cars))
            while not self.car_arrived(0):
                car0_state = self.cars[0].get_state(self.intersection, 0)
                # for each car, check collision with car 0
                for i in not_arrived_cars:
                    if self.cars_intersect(self.cars[0], self.cars[i]):
                        if car0_state not in self.collisions:
                            self.collisions[car0_state] = []
                        car1_state = self.cars[i].get_state(intersection, 0)
                        if car1_state not in self.collisions[car0_state] and car1_state is not car0_state:
                            self.collisions[car0_state].append(car1_state)

                    # move car i one step forward
                    self.advance_car(i)

                    if self.car_arrived(i):
                        not_arrived_cars.remove(i)

                # all cars terminated -> move car 0 1 step ahead and reset all other cars
                if len(not_arrived_cars) is 0:
                    self.advance_car(0)

                    not_arrived_cars = list(range(1, self.num_cars))
                    for i in not_arrived_cars:
                        self.cars[i].node_idx = self.start_nodes[i]
                        self.cars[i].route_segment_idx = 0

    def extend_to_4_quadrants(self):
        nodes_per_quadrant = self.intersection.quadrant_lookup[0]
        total_nodes = self.intersection.quadrant_lookup[-1]
        for i in range(1, 4):
            keys = [key for key in self.collisions]
            for key in keys:
                self.collisions[(key + i * nodes_per_quadrant) %
                                total_nodes] = [(entry + i * nodes_per_quadrant) %
                                                total_nodes for entry in self.collisions[key]]

    def plot_results(self):
        plt.rcParams.update({'figure.max_open_warning': 0})
        for point_idx in self.collisions:
            plt.figure(point_idx)
            plot_street(plt)
            for collision_point in self.collisions[point_idx]:
                self.cars[0].plot_pose(plt, intersection, intersection.get_point(collision_point))
            self.cars[0].plot_pose(plt, intersection, intersection.get_point(point_idx), 'b-')
        plt.show()


intersection = Intersection(intersection_params)
collision_table = CollisionTable(intersection)
collision_table.calculate_collision_table()
collision_table.extend_to_4_quadrants()
collision_table.plot_results()
