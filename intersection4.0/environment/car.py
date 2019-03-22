"""Car class

This module contains the car class, allowing to track their state (position in the intersection, velocity, planned destination) and apply actions (accelerate, keep speed, decelerate).

author: Christof Dubs
"""
import numpy as np
from definitions import SectionIndex, Target, Action
from rot_2d import rot_z


class CarParams:
    def __init__(self):
        self.width = 2.0
        self.length = 4.0
        self.back_axis = 3.0
        self.speed_increment = 2
        self.min_speed = 0
        self.max_speed = 20
        self.max_speed_long_turn = 10
        self.max_speed_sharp_turn = 4

    def saturate_vel(self, vel):
        if vel > self.max_speed:
            return self.max_speed

        if vel < self.min_speed:
            return self.min_speed

        return vel


class Car:
    def __init__(self, param):
        self.param = param
        self.node_idx = -1
        self.is_spawned = False
        self.progress = 0

    def is_active(self):
        return self.node_idx >= 0

    def spawn(self, start_quadrant, target, init_vel, intersection):
        self.vel = self.param.saturate_vel(init_vel)
        self.route = [[start_quadrant, SectionIndex.BEFORE_INTERSECTION]]

        if target is Target.GO_STRAIGHT:
            middle_section = SectionIndex.IN_INTERSECTION_STRAIGHT
            end_quadrant = start_quadrant
        elif target is Target.TURN_RIGHT:
            middle_section = SectionIndex.IN_INTERSECTION_RIGHT
            end_quadrant = (start_quadrant + 3) % 4
        elif target is Target.TURN_LEFT:
            middle_section = SectionIndex.IN_INTERSECTION_LEFT
            end_quadrant = (start_quadrant + 1) % 4
        else:
            raise ValueError("invalid target: {}".format(target))

        self.route.append([start_quadrant, middle_section])
        self.route.append([end_quadrant, SectionIndex.AFTER_INTERSECTION])

        self.node_idx = 0
        self.route_segment_idx = 0
        self.route_segment_lookup = [intersection.quadrants[e[0]
                                                            ].segments[e[1]].num_points for e in self.route]

        self.outline = self.get_outline

    def set_action(self, action):
        if not self.is_active():
            return 0

        if action is Action.KEEP_SPEED:
            self.future_vel = self.vel
        elif action is Action.ACCELERATE:
            self.future_vel = self.param.saturate_vel(self.vel + self.param.speed_increment)
        elif action is Action.DECELERATE:
            self.future_vel = self.param.saturate_vel(self.vel - self.param.speed_increment)
        else:
            raise ValueError("invalid action: {}".format(action))

        self.progress = int((self.future_vel + self.vel) / 2)
        self.calculate_future_indices()
        return np.sqrt(self.future_vel)

    def execute_action(self):
        self.vel = self.future_vel
        self.node_idx = self.future_node_idx
        self.route_segment_idx = self.future_route_segment_idx

    def calculate_future_indices(self):
        self.future_node_idx = self.node_idx + self.progress
        self.future_route_segment_idx = self.route_segment_idx
        self.future_route_segment_idx, self.future_node_idx = self.wrap_indices(
            self.future_route_segment_idx, self.future_node_idx)

    def wrap_indices(self, route_segment_idx, node_idx):
        while (node_idx >= self.route_segment_lookup[route_segment_idx]):
            node_idx -= self.route_segment_lookup[route_segment_idx]
            route_segment_idx += 1
            if route_segment_idx > 2:
                node_idx = -1
                break
        return [route_segment_idx, node_idx]

    def get_state(self, intersection, viewer_quadrant):
        if not self.is_active():
            return -1

        q, s = self.route[self.route_segment_idx]
        return intersection.get_point_idx((q - viewer_quadrant) % 4, s, self.node_idx)

    def get_interpolated_pose(self, intersection, alpha):
        if not self.is_active():
            return np.array([np.inf, np.inf, 0])

        q, s = self.route[self.route_segment_idx]
        segment = intersection.quadrants[q].segments[s]
        length = segment.param.start_offset + segment.param.step_size * \
            (self.node_idx + alpha * ((1 - alpha * 0.5) * self.vel + alpha * 0.5 * self.future_vel))
        if length < segment.length:
            return segment.calc_point(length)

        if self.route_segment_idx >= 2:
            return np.array([np.inf, np.inf, 0])

        q, s = self.route[self.route_segment_idx + 1]
        segment2 = intersection.quadrants[q].segments[s]
        return segment2.calc_point(length - segment.length)

    def get_pose(self, intersection):
        if not self.is_active():
            return np.array([np.inf, np.inf, 0])

        q, s = self.route[self.route_segment_idx]
        return intersection.quadrants[q].segments[s].get_point(self.node_idx)

    def get_outline(self):
        half_width = self.param.width / 2
        front = self.param.back_axis
        back = self.param.back_axis - self.param.length
        return np.array([[front, back, back, front], [
                        half_width, half_width, -half_width, -half_width]])

    def get_transformed_outline(self, pose):
        return np.dot(rot_z(pose[2]), self.get_outline()) + np.outer(pose[0:2], np.ones([1, 4]))

    def plot(self, plt, intersection, style='r-'):
        shape = self.get_transformed_outline(self.get_pose(intersection))
        plt.plot(shape[0, :], shape[1, :], style)

    def plot_pose(self, plt, intersection, pose, style='r-'):
        shape = self.get_transformed_outline(pose)
        plt.plot(shape[0, :], shape[1, :], style)

    def plot_interpolated(self, plt, intersection, alpha):
        shape = self.get_transformed_outline(self.get_interpolated_pose(intersection, alpha))
        plt.plot(shape[0, :], shape[1, :], 'r-')
