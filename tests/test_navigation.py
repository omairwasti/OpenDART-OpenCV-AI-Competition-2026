import numpy as np
import pytest
from numpy.testing import assert_allclose
from opendart.camera import Camera, Detection, pointing_rotation, crop_intrinsics, calibrated_valid_mask
from opendart.navigation import BearingEKF
from opendart.model import Environment


def test_camera_jacobian_and_range_nullspace():
    c = Camera()
    x = np.array([20., -15., 10000.])
    r = pointing_rotation([100., 30., 10000.])
    h = c.jacobian(x, r)[:, :3]
    eps = .01
    fd = np.column_stack([(c.project(x+eps*np.eye(3)[i], r)
                           - c.project(x-eps*np.eye(3)[i], r))/(2*eps) for i in range(3)])
    assert_allclose(h, fd, atol=1e-8)
    assert_allclose(h @ x, np.zeros(2), atol=1e-10)
    assert_allclose(c.project(2*x, r), c.project(x, r))
    assert_allclose(r @ r.T, np.eye(3), atol=1e-15)
    assert_allclose(np.linalg.det(r), 1)


def test_crop_and_special_pixel_masks():
    c = Camera()
    crop = crop_intrinsics(c, 200, 100, 512, 512)
    assert_allclose(crop.project([1, 1, 1e4], np.eye(3)),
                    c.project([1, 1, 1e4], np.eye(3))-[200, 100])
    arr = np.array([-1e10, -1e9, -1e8, -0.01, 0., 1., 1e8, 1e9, 1e10, np.nan])
    assert calibrated_valid_mask(arr).tolist() == [False, False, False, True, True, True,
                                                   False, False, False, False]


def test_ekf_gating_timestamp_and_covariance():
    c = Camera()
    x = np.array([0., 0., 1e5, 0., 0., -100.])
    n = BearingEKF(x, np.diag([100., 100., 1e6, 1., 1., 1.]))
    z = c.project(x[:3], np.eye(3))
    assert n.update(Detection(0., z+[.1, -.1], np.eye(2)*.1), c, np.eye(3), 1)
    assert not n.update(Detection(0., z+[100., 100.], np.eye(2)*.1), c, np.eye(3), 1)
    assert not n.update(Detection(-1., z, np.eye(2)), c, np.eye(3), 1)
    assert not n.update(Detection(0., z, np.eye(2), target_id=0), c, np.eye(3), 1)
    assert np.linalg.eigvalsh(n.P).min() > 0
    with pytest.raises(ValueError):
        n.update(Detection(0., z, -np.eye(2)), c, np.eye(3), 1)


def test_bearing_does_not_collapse_axial_range_variance():
    c = Camera()
    n = BearingEKF([0, 0, 1e5, 0, 0, -100], np.diag([100, 100, 1e6, 1, 1, 1]))
    before = n.P[2, 2]
    n.update(Detection(0., [511.5, 511.5], np.eye(2)*.1), c, np.eye(3), 1)
    assert_allclose(n.P[2, 2], before)
    assert n.P[0, 0] < 100


def test_predict_matches_applied_acceleration_and_grows_uncertainty():
    e = Environment(solar_tide=False, srp=False)
    x = np.array([0., 0., 1e6, 0., 0., -100.])
    n = BearingEKF(x, np.eye(6), acceleration_psd=1e-4)
    expected = e.relative_drift(0., x, 500.)-np.array([.02, 0, 0])
    n.predict(.01, np.array([10., 0, 0]), 500., 220., e, 1)
    assert_allclose((n.x[3:]-x[3:])/.01, expected, atol=1e-8)
    assert np.trace(n.P) > 6.
