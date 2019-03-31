import unittest

import context

import numpy as np

from intersection4.environments.base.param import intersection_params
from intersection4.environments.base.intersection import Intersection


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


if __name__ == '__main__':
    unittest.main()
