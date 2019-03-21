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
        self.length = param.length
        self.points = np.zeros([3, self.num_points])

    def get_point(self, node_idx):
        if node_idx < 0 or node_idx >= self.num_points:
            return None
        return self.points[:, node_idx]


class LineSegment(Segment):
    def __init__(self, start, param):
        super().__init__(param)
        self.param = param
        self.start = start
        self.dir = np.array([np.cos(start[2]), np.sin(start[2]), 0])
        for i in range(param.num_points):
            self.points[:, i] = self.calc_point(param.start_offset + i * param.step_size)

    def calc_point(self, l):
        return self.start + l * self.dir


class TurnSegment(Segment):
    def __init__(self, start, param, right=True):
        super().__init__(param)
        self.param = param

        self.angle_dir = -1 if right else 1
        self.arc_start_angle = start[2] - self.angle_dir * np.pi / 2

        arc_start_vec = param.radius * \
            np.array([np.cos(self.arc_start_angle), np.sin(self.arc_start_angle), 0])
        self.arc_center = start - arc_start_vec

        start_angle = param.start_offset / param.radius
        angle_incr = param.step_size / param.radius

        for i in range(param.num_points):
            self.points[:, i] = self.calc_point(param.start_offset + i * param.step_size)

    def calc_point(self, l):
        angle_delta = self.angle_dir * l / self.param.radius
        angle = self.arc_start_angle + angle_delta
        return self.arc_center + \
            np.array([self.param.radius * np.cos(angle), self.param.radius * np.sin(angle), angle_delta])


class LeftTurn(Segment):
    def __init__(self, start, straight_length, param):
        self.param = param
        self.segments = [None] * 3
        rot = rot_z(start[2])
        init_line_param = LineSegmentParam(straight_length, param.step_size, param.start_offset)
        self.segments[0] = LineSegment(start, init_line_param)

        curve_param = CurveSegmentParam(
            param.radius,
            param.step_size,
            init_line_param.length_overflow)
        curve_start = start + np.append(np.dot(rot, np.array([straight_length, 0])), 0)
        self.segments[1] = TurnSegment(curve_start, curve_param, False)

        end_line_param = LineSegmentParam(
            straight_length,
            param.step_size,
            curve_param.length_overflow)

        end_line_start = curve_start + \
            np.append(np.dot(rot, np.array([param.radius, param.radius])), np.pi / 2)
        self.segments[2] = LineSegment(end_line_start, end_line_param)

        self.points = self.segments[1].points

        if (init_line_param.num_points > 0):
            self.points = np.concatenate([self.segments[0].points, self.points], axis=1)

        if (end_line_param.num_points > 0):
            self.points = np.concatenate([self.points, self.segments[2].points], axis=1)

        self.num_points = np.shape(self.points)[1]
        self.segment_lookup = np.cumsum([s.length for s in self.segments])
        self.length = self.segment_lookup[-1]

    def calc_point(self, l):
        segment_idx = np.searchsorted(self.segment_lookup, l + 1)
        if segment_idx > 0:
            l -= self.segment_lookup[segment_idx - 1]
        return self.segments[segment_idx].calc_point(l)


class IntersectionQuadrant:
    def __init__(self, param, quadrant):
        self.segments = [None] * SI.NUM_SEGMENTS

        start_line_start = self.rotate_to_quadrant(np.array(
            [param.intersection_dim + param.straight_length, param.lane_width * 0.5, np.pi]), quadrant)
        start_line_param = LineSegmentParam(param.straight_length, param.step_size, 0)

        self.segments[SI.BEFORE_INTERSECTION] = LineSegment(start_line_start, start_line_param)

        middle_line_start = self.rotate_to_quadrant(
            np.array([param.intersection_dim, param.lane_width * 0.5, np.pi]), quadrant)
        middle_line_param = LineSegmentParam(
            2 * param.intersection_dim,
            param.step_size,
            start_line_param.length_overflow)

        self.segments[SI.IN_INTERSECTION_STRAIGHT] = LineSegment(
            middle_line_start, middle_line_param)

        end_line_start = self.rotate_to_quadrant(
            np.array([-param.intersection_dim, param.lane_width * 0.5, np.pi]), quadrant)
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

    def rotate_to_quadrant(self, point, quadrant):
        angle = (quadrant % 4) * np.pi / 2
        rot = rot_z(angle)
        return np.append(np.dot(rot, point[:2]), point[2] + angle)

    def get_point_idx(self, segment_idx, node_idx):
        if segment_idx == 0:
            return node_idx

        return self.segment_lookup[segment_idx - 1] + node_idx

    def get_point_from_idx(self, point_idx):
        segment_idx = np.searchsorted(self.segment_lookup, point_idx + 1)
        node_idx = point_idx if segment_idx == 0 else point_idx - \
            self.segment_lookup[segment_idx - 1]
        return self.get_point(segment_idx, node_idx)

    def get_point(self, segment_idx, node_idx):
        return self.segments[segment_idx].get_point(node_idx)

    def get_all_points(self):
        return [self.get_point(a, b) for a in range(5) for b in range(self.segments[a].num_points)]


class Intersection:
    def __init__(self, param):
        self.quadrants = [IntersectionQuadrant(param, i) for i in range(4)]
        self.quadrant_lookup = np.cumsum([q.segment_lookup[-1] for q in self.quadrants])

    def get_point_idx(self, quadrant, segment, node_idx):
        quadrant_node_idx = self.quadrants[quadrant].get_point_idx(segment, node_idx)
        if quadrant == 0:
            return quadrant_node_idx

        return self.quadrant_lookup[quadrant - 1] + quadrant_node_idx

    def get_point(self, point_idx):
        quadrant_idx = np.searchsorted(self.quadrant_lookup, point_idx + 1)
        seg_point_idx = point_idx if quadrant_idx == 0 else point_idx - \
            self.quadrant_lookup[quadrant_idx - 1]
        return self.quadrants[quadrant_idx].get_point_from_idx(seg_point_idx)

    def plot_points(self, plt):
        points = [self.quadrants[i].get_all_points() for i in range(4)]

        plot_street(plt)
        styles = ['ro', 'b*', 'g+', 'y.']
        for i in range(4):
            plt.plot([p[0] for p in points[i]], [p[1] for p in points[i]], styles[i])
