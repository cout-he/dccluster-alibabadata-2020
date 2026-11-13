"""Independent, hand-calculated checks for the numerical analysis core."""
import unittest
import numpy as np
from workload_analysis import event_sweep, weighted_summary, arrival_counts


class AnalysisMathTests(unittest.TestCase):
    def test_half_open_intervals_and_idle_time(self):
        t, v, w = event_sweep(np.array([0., 2., 4.]), np.array([2., 2., 1.]),
                              np.array([[3.], [7.], [2.]]), 6.)
        np.testing.assert_allclose(t, [0, 2, 4, 5])
        np.testing.assert_allclose(v[:, 0], [3, 7, 2, 0])
        np.testing.assert_allclose(w, [2, 2, 1, 1])
        self.assertAlmostEqual(float((v[:, 0] * w).sum()), 22)

    def test_overlaps_and_time_weighted_quantiles(self):
        t, v, w = event_sweep(np.array([0., 1.]), np.array([4., 1.]),
                              np.array([[1.], [9.]]), 10.)
        s = weighted_summary(t, v[:, 0], w)
        self.assertAlmostEqual(s['mean'], 1.3)
        self.assertEqual(s['p50'], 0)
        self.assertEqual(s['p95'], 10)
        self.assertEqual(s['peak_seconds'], 1)

    def test_periodic_carry_in(self):
        t, v, w = event_sweep(np.array([9.]), np.array([3.]),
                              np.array([[2.]]), 10., periodic=True)
        self.assertAlmostEqual(float((v[:, 0] * w).sum()), 6)
        self.assertAlmostEqual(v[0, 0], 2)

    def test_arrivals_include_empty_bins(self):
        np.testing.assert_array_equal(arrival_counts(np.array([0., 4., 5., 19.]), 5, 20),
                                      [2, 1, 0, 1])


if __name__ == '__main__':
    unittest.main()
