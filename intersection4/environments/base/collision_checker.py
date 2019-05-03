"""Collision checker

This module contains a class to check collisions between multiple cars

author: Christof Dubs
"""
import numpy as np
from intersection import Intersection
from car import Car, CarParams
from definitions import Target
from collision_table import CollisionTable


NO_COLLISION = 0
COLLISION = 1
UNDETERMINED = 2

OUTSIDE_INTERSECTION = 0
INSIDE_INTERSECTION = 1
TRANSITIONING_INTERSECTION = 2


class CollisionChecker:
    def __init__(self, intersection):
        self.intersection = intersection
        self.collision_table = CollisionTable(intersection)
        self.collision_table.calculate_collision_table()
        self.collision_table.extend_to_4_quadrants()

    def check_collisions(self, cars):
        car_tables = [self.create_car_segment_and_node_time_table(car) for car in cars]

        car_checklist = self.create_car_checklist(cars)

        collision_pairs = []
        num_cars = len(cars)

        for i in range(num_cars):
            for j in car_checklist[i]:
                if self.definitely_no_segment_collision(car_tables[i][0], car_tables[j][0]):
                    continue

                no_collision_dist = cars[i].param.length / self.intersection.param.step_size

                lookup_result = self.check_node_table(
                    car_tables[i][1], car_tables[j][1], no_collision_dist)

                if lookup_result == COLLISION:
                    collision_pairs.append([i, j])

        return collision_pairs

    def check_node_table(self, car_1_node_table, car_2_node_table, no_collision_dist):
        car_2_ref_node = 0

        num_car_1_nodes = len(car_1_node_table)
        car_1_collisions = [[] for i in range(num_car_1_nodes)]

        for car_1_ref_node in range(1, num_car_1_nodes):
            while car_2_ref_node < len(car_2_node_table):

                car_1_collisions[car_1_ref_node].append(
                    self.check_node_collision(
                        car_1_node_table[car_1_ref_node],
                        car_2_node_table[car_2_ref_node],
                        no_collision_dist))

                if car_2_node_table[car_2_ref_node][0] == car_1_node_table[car_1_ref_node][0]:
                    break
                if car_2_node_table[car_2_ref_node][0] > car_1_node_table[car_1_ref_node][0]:
                    car_2_ref_node -= 1
                    break

                car_2_ref_node += 1

        # todo: check for 2 consecutive True entries, and add more exact checks if
        # there is just 1 consecutive True entry
        for entries in car_1_collisions:
            if any(x for x in entries):
                return COLLISION

        return NO_COLLISION

    def check_node_collision(self, node_table_entry_1, node_table_entry_2, no_collision_dist):
        node_1_idx = node_table_entry_1[1]
        node_2_idx = node_table_entry_2[1]

        # if any car outside intersection: check simple node distance (note: this
        # only works because two different segments outside intersection's node
        # indices are far enough apart
        if node_table_entry_1[2] == OUTSIDE_INTERSECTION or node_table_entry_2[2] == OUTSIDE_INTERSECTION:
            return np.abs(node_1_idx - node_2_idx) < no_collision_dist

        # if both cars inside intersection, consult collision table
        return node_2_idx in self.collision_table.collisions[node_1_idx]

    def definitely_no_segment_collision(self, seg_table_1, seg_table_2):
        for entry_1 in seg_table_1:
            for entry_2 in seg_table_2:
                if self.segments_overlap(entry_1, entry_2):
                    return False
        return True

    def segments_overlap(self, seg_1, seg_2):
        # check time overlap
        if seg_1[0][0] > seg_2[0][1] or seg_2[0][0] > seg_1[0][1]:
            return False

        if seg_1[3] and seg_2[3]:
            # both in intersection
            return True

        else:
            # at least one not in intersection
            if seg_1[1] == seg_2[1] and seg_1[2] == seg_2[2]:
                return True

            return False

    def create_car_checklist(self, cars):
        num_cars = len(cars)
        car_checklist = [[] for i in range(num_cars)]
        for i in range(num_cars):
            if not cars[i].is_active():
                continue
            for j in range(i + 1, num_cars):
                if self.routes_overlap(cars[i], cars[j]):
                    car_checklist[i].append(j)

        return car_checklist

    def routes_overlap(self, car_1, car_2):
        quad_diff = (car_2.route[0][0] - car_1.route[0][0]) % 4

        if quad_diff == 0:
            return True

        car_1_target = car_1.target
        car_2_target = car_2.target

        if car_1_target == Target.GO_STRAIGHT:
            return quad_diff != 2 or car_2_target == Target.TURN_LEFT

        if car_1_target == Target.TURN_RIGHT:
            if quad_diff == 1:
                return False
            if quad_diff == 2:
                return car_2_target == Target.TURN_LEFT
            if quad_diff == 3:
                return car_2_target == Target.GO_STRAIGHT
            assert(False)

        if car_1_target == Target.TURN_LEFT:
            if quad_diff == 1:
                return car_2_target != Target.TURN_RIGHT
            if quad_diff == 2:
                return car_2_target != Target.TURN_LEFT
            return True
        assert(False)

    def calculate_time_vector(self, car):
        if car.vel == car.future_vel:
            if car.vel == 0:
                return [0]
            return list(np.linspace(0, 1, car.vel + 1))
        delta_vel = car.future_vel - car.vel
        return [(-car.vel + np.sqrt(car.vel**2 + 2 * delta_vel * i)) / delta_vel
                for i in range(int((car.future_vel + car.vel) / 2) + 1)]

    def create_car_segment_and_node_time_table(self, car):
        if not car.is_active():
            return [], []

        route_segment_idx = car.route_segment_idx
        q, s = car.route[route_segment_idx]
        n = car.node_idx

        time_vec = self.calculate_time_vector(car)
        point_idx = self.intersection.get_point_idx(q, s, n)
        if route_segment_idx == 1:
            status = INSIDE_INTERSECTION
        elif point_idx in self.collision_table.collisions:
            status = TRANSITIONING_INTERSECTION
        else:
            status = OUTSIDE_INTERSECTION

        time_table = [[time_vec[0], point_idx, status]]
        segment_table = [[[time_vec[0], None], q, s, status != OUTSIDE_INTERSECTION]]

        # special case of car in rest
        if len(time_vec) == 1:
            time_table.append([1, point_idx, status])
            segment_table[-1][0][-1] = 1
            return segment_table, time_table

        for t in time_vec[1:]:
            new_route_segment_idx, n = car.wrap_indices(route_segment_idx, n + 1)
            if new_route_segment_idx != route_segment_idx:
                if new_route_segment_idx > 2:
                    segment_table[-1][0][-1] = t
                    return segment_table, time_table

                route_segment_idx = new_route_segment_idx
                q, s = car.route[route_segment_idx]
                segment_table[-1][0][-1] = t
                segment_table.append([[t, None], q, s, False])

            point_idx = self.intersection.get_point_idx(q, s, n)
            if route_segment_idx == 1:
                status = INSIDE_INTERSECTION
            elif point_idx in self.collision_table.collisions:
                status = TRANSITIONING_INTERSECTION
            else:
                status = OUTSIDE_INTERSECTION
            time_table.append([t, point_idx, status])

            if not segment_table[-1][3]:
                segment_table[-1][3] = status != OUTSIDE_INTERSECTION

        segment_table[-1][0][-1] = time_vec[-1]
        return segment_table, time_table
