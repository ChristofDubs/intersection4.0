"""2 car environment

This module contains a class to simulate an episode with two controlled cars

author: Christof Dubs
"""
import context
import numpy as np
import matplotlib.pyplot as plt
from environments.base.intersection import Intersection
from environments.base.car import Car, CarParams
from environments.base.param import intersection_params, plot_street
from environments.base.definitions import Target
from environments.base.collision_checker import CollisionChecker


class TwoCarEnv:
    def __init__(self):
        self.intersection = Intersection(intersection_params)
        self.collision_checker = CollisionChecker(self.intersection)
        self.cars = [Car(CarParams()) for _ in range(2)]

    def reset(self):
        self.cars[0].spawn(0, Target.GO_STRAIGHT, 6, self.intersection)
        self.cars[1].spawn(1, Target.GO_STRAIGHT, 6, self.intersection)
        self.collision_detected = False
        return self.get_observation()

    def get_observation(self):
        observation = []
        for car in self.cars:
            observation.extend([car.get_state(self.intersection, 0), car.vel])
        return observation

    def step(self, action_pair):
        for i, car in enumerate(self.cars):
            car.set_action(action_pair[i])

        self.check_collisions()
        reward = self.calculate_reward()

        for car in (self.cars):
            if car.is_active():
                car.execute_action()

        observation = self.get_observation()

        done = self.collision_detected or all([not car.is_active() for car in self.cars])

        return [observation, reward, done, None]

    def check_collisions(self):
        self.collision_detected = not not self.collision_checker.check_collisions(self.cars)

    def calculate_reward(self):
        reward = 0
        for car in (self.cars):
            if car.is_active():
                reward -= 1
        if self.collision_detected:
            reward -= 10000
        return reward

    def render(self):
        plt.figure(0)
        plt.cla()
        plot_street(plt)
        style = 'r-' if self.collision_detected else 'g-'
        for car in self.cars:
            car.plot(plt, self.intersection, style)
        plt.show(block=False)
        plt.pause(0.02)
