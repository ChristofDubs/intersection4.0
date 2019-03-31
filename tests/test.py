import unittest

import context

import numpy as np

from intersection4.environments.base.param import intersection_params
from intersection4.environments.base.intersection import Intersection
from intersection4.environments.base.collision_table import CollisionTable


class TestIntersection(unittest.TestCase):
    def test_point_index(self):
        intersection = Intersection(intersection_params)
        ref_idx = 0
        for q in range(3):
            for s in range(5):
                for p in range(intersection.quadrants[q].segments[s].num_points):
                    point_idx = intersection.get_point_idx(q, s, p)
                    self.assertEqual(point_idx, ref_idx)
                    ref_idx += 1

                    point = intersection.quadrants[q].get_point(s, p)
                    point_ref = intersection.get_point(point_idx)
                    self.assertEqual(point_ref[0], point[0])
                    self.assertEqual(point_ref[1], point[1])
                    self.assertEqual(point_ref[2], point[2])


class TestCollisionTable(unittest.TestCase):
    def test_consistency(self):
        intersection = Intersection(intersection_params)
        ct = CollisionTable(intersection)
        ct.calculate_collision_table()
        ct.extend_to_4_quadrants()
        for key in ct.collisions:
            for el in ct.collisions[key]:
                self.assertTrue(key in ct.collisions[el])


if __name__ == '__main__':
    unittest.main()
