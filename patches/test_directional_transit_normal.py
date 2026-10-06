"""Directional transit labels crossings along the aperture normal, not by the crossing step's y-velocity
(2026-09-25). At a yawed gate the drone can cross moving mostly along x, where the step's dy is a few mm of either
sign; that used to flip correct passes to wrong-way passes and back."""
import numpy as np

from falsify.safety.posthoc import check_directional_transit

# The left gate: posts at (0.59, 1.01) and (1.165, 0.359), normal ~(0.75, 0.66). A correct left transit moves +y.
CORNERS = np.array([[0.59, 1.01, 0.2], [1.165, 0.359, 0.2], [1.165, 0.359, 1.896], [0.59, 1.01, 1.896]])
AABB = (np.zeros(3), np.ones(3))
MID = CORNERS[:2, :2].mean(0)
N = np.array([0.749, 0.663])


def path(*xy):
    return np.array([[x, y, 1.5] for x, y in xy])


def judge(P):
    return check_directional_transit(P, *AABB, expected_dy_sign=+1, aperture_corners=CORNERS)


def test_crossing_along_x_with_negative_dy_is_correct():
    a = MID - np.array([0.06, 0.0])
    r = judge(path(a, a + np.array([0.12, -0.002])))       # moves +x through the gate, dy slightly negative
    assert (r.correct_crossings, r.wrong_crossings, r.first_correct_step) == (1, 0, 0)


def test_backing_out_with_positive_dy_is_wrong():
    a = MID + np.array([0.06, 0.0])
    r = judge(path(a, a + np.array([-0.12, 0.002])))       # moves -x back through the gate, dy slightly positive
    assert (r.correct_crossings, r.wrong_crossings) == (0, 1)


def test_crossing_outside_the_opening_is_ignored():
    off = MID + 0.6 * np.array([-N[1], N[0]])               # 0.6 m along the gate from the centre: past a post
    r = judge(path(off - 0.05 * N, off + 0.05 * N))
    assert (r.correct_crossings, r.wrong_crossings) == (0, 0)
