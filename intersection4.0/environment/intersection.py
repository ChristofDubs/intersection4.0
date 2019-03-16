"""Intersection class

This module contains the classes for expressing positions as discretized points in the intersection.

author: Christof Dubs
"""
import numpy as np
from param import intersection_params, plot_street
from definitions import SectionIndex as SI
from rot_2d import rot_z


class SegmentParam:
    def __init__(self, length, step_size, start_offset=0):
        assert(start_offset < step_size)
        self.length = length
        self.step_size = step_size
        self.start_offset = start_offset
        self.num_points = int((length - start_offset) / step_size) + 1
        self.length_overflow = np.mod(self.step_size - (length - start_offset), step_size)


class LineSegmentParam(SegmentParam):
    def __init__(self, length, step_size, start_offset):
        super().__init__(length, step_size, start_offset)


class CurveSegmentParam(SegmentParam):
    def __init__(self, radius, step_size, start_offset):
        self.radius = radius
        length = radius * np.pi / 2
        super().__init__(length, step_size, start_offset)


class Segment:
    def __init__(self, param):
        self.num_points = param.num_points
        self.points = np.zeros([3, self.num_points])

    def get_point(self, node_idx):
        if node_idx < 0 or node_idx >= self.num_points:
            return None
        return self.points[:, node_idx]


class LineSegment(Segment):
    def __init__(self, start, param):
        super().__init__(param)
        self.param = param
        self.dir = np.array([np.cos(start[2]), np.sin(start[2]), 0])
        for i in range(param.num_points):
            self.points[:, i] = start + (param.start_offset + i * param.step_size) * self.dir


class TurnSegment(Segment):
    def __init__(self, start, param, right=True):
        super().__init__(param)
        start_angle = param.start_offset / param.radius
        angle_incr = param.step_size / param.radius

        dir = 1 if right else -1

        arc_center = start + np.array([0, dir * param.radius, 0])
        for i in range(param.num_points):
            angle = start_angle + i * angle_incr
            self.points[:, i] = arc_center + \
                np.array([-param.radius * np.sin(angle), -dir * param.radius * np.cos(angle), -dir * angle])


class LeftTurn(Segment):
    def __init__(self, start, straight_length, param):
        init_line_param = LineSegmentParam(straight_length, param.step_size, param.start_offset)

        curve_param = CurveSegmentParam(
            param.radius,
            param.step_size,
            init_line_param.length_overflow)
        curve_start = start + np.array([-straight_length, 0, 0])
        curve = TurnSegment(curve_start, curve_param, False)

        end_line_param = LineSegmentParam(
            straight_length,
            param.step_size,
            curve_param.length_overflow)

        self.points = curve.points
        if (init_line_param.num_points > 0):
            init_line = LineSegment(start, init_line_param)
            self.points = np.concatenate([init_line.points, self.points], axis=1)

        if (end_line_param.num_points > 0):
            end_line_start = start + \
                np.array([-straight_length - param.radius, -param.radius, np.pi / 2])
            end_line = LineSegment(end_line_start, end_line_param)
            self.points = np.concatenate([self.points, end_line.points], axis=1)

        self.num_points = np.shape(self.points)[1]


class IntersectionQuadrant:
    def __init__(self, param):
        self.segments = [None] * SI.NUM_SEGMENTS

        start_line_start = np.array(
            [param.intersection_dim + param.straight_length, param.lane_width * 0.5, np.pi])
        start_line_param = LineSegmentParam(param.straight_length, param.step_size, 0)

        self.segments[SI.BEFORE_INTERSECTION] = LineSegment(start_line_start, start_line_param)

        middle_line_start = np.array([param.intersection_dim, param.lane_width * 0.5, np.pi])
        middle_line_param = LineSegmentParam(
            2 * param.intersection_dim,
            param.step_size,
            start_line_param.length_overflow)

        self.segments[SI.IN_INTERSECTION_STRAIGHT] = LineSegment(
            middle_line_start, middle_line_param)

        end_line_start = np.array([-param.intersection_dim, param.lane_width * 0.5, np.pi])
        end_line_param = LineSegmentParam(
            param.straight_length,
            param.step_size,
            middle_line_param.length_overflow)

        self.segments[SI.AFTER_INTERSECTION] = LineSegment(end_line_start, end_line_param)

        right_turn_param = CurveSegmentParam(
            param.border_radius + param.lane_width * 0.5,
            param.step_size,
            start_line_param.length_overflow)

        self.segments[SI.IN_INTERSECTION_RIGHT] = TurnSegment(
            middle_line_start, right_turn_param, True)

        left_turn_param = CurveSegmentParam(
            param.border_radius +
            param.lane_width *
            1.5 -
            param.left_turn_straight_length,
            param.step_size,
            start_line_param.length_overflow)

        self.segments[SI.IN_INTERSECTION_LEFT] = LeftTurn(
            middle_line_start, param.left_turn_straight_length, left_turn_param)

        self.segment_lookup = np.cumsum([s.num_points for s in self.segments])
        self.num_points = self.segment_lookup[-1]

    def get_point_idx(self, segment_idx, node_idx):
        if segment_idx == 0:
            return node_idx

        return self.segment_lookup[segment_idx - 1] + node_idx

    def get_point(self, segment_idx, node_idx):
        return self.segments[segment_idx].get_point(node_idx)

    def get_all_points(self):
        return [self.get_point(a, b) for a in range(5) for b in range(self.segments[a].num_points)]


class Intersection:
    def __init__(self, param):
        self.quadrants = [IntersectionQuadrant(param) for i in range(4)]
        for i in range(1, 4):
            rot_angle = i * np.pi / 2
            rot_mat = rot_z(rot_angle)
            for segment in self.quadrants[i].segments:
                segment.points[0:2, :] = np.dot(rot_mat, segment.points[0:2, :])
                segment.points[2, :] += rot_angle

        self.quadrant_lookup = np.cumsum([q.segment_lookup[-1] for q in self.quadrants])

    def get_point_idx(self, quadrant, segment, node_idx):
        quadrant_node_idx = self.quadrants[quadrant].get_point_idx(segment, node_idx)
        if quadrant == 0:
            return quadrant_node_idx

        return self.quadrant_lookup[quadrant - 1] + quadrant_node_idx

    def plot_points(self, plt):
        points = [self.quadrants[i].get_all_points() for i in range(4)]

        plot_street(plt)
        styles = ['ro', 'b*', 'g+', 'y.']
        for i in range(4):
            plt.plot([p[0] for p in points[i]], [p[1] for p in points[i]], styles[i])
