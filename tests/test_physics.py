import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from opendart.model import Binary, Environment, jacobi, MU_SUN
from opendart.guidance import allocate_force, Guidance
from opendart.simulation import Scenario, environment, advance


def test_binary_kepler_and_barycenter():
    b = Binary()
    p, v, a = b.states(4321.)
    assert_allclose(b.mu @ p, np.zeros(3), atol=1e-11)
    assert_allclose(b.mu @ v, np.zeros(3), atol=1e-14)
    d = p[1]-p[0]
    assert_allclose(a[1]-a[0], -sum(b.mu)*d/np.linalg.norm(d)**3, atol=1e-18)
    assert_allclose(sum(b.mu), b.n**2*b.separation_m**3)


def test_gravity_gradient_and_relative_sign():
    e = Environment(solar_tide=False, srp=False)
    pos = np.array([4000., 3000., 2000.])
    h = .1
    numerical = np.column_stack([(e.gravity(13., pos+h*np.eye(3)[i])
                                 - e.gravity(13., pos-h*np.eye(3)[i]))/(2*h)
                                for i in range(3)])
    assert_allclose(e.gravity_gradient(13., pos), numerical, rtol=1e-8, atol=1e-17)
    x = np.r_[e.binary.states(13.)[0][1]-pos, [1., 2., 3.]]
    # d[aT-g(rT-r)]/dr = +Dg, not -Dg
    assert_allclose(e.relative_jacobian(13., x, 500.)[3:, :3], numerical,
                    rtol=1e-7, atol=1e-17)


def test_tide_is_differential_and_srp_points_away():
    e = Environment(solar_tide=True, srp=False)
    pos = np.array([1e5, 0., 0.])
    tide = e.acceleration(0., pos, 500)-e.gravity(0., pos)
    s = e.sun_position_m
    linear = MU_SUN/np.linalg.norm(s)**3 * np.array([2e5, 0., 0.])
    assert_allclose(tide, linear, rtol=2e-6, atol=1e-18)
    e.solar_tide, e.srp = False, True
    pos = np.array([1e4, 0., 1e4])
    srp = e.acceleration(0., pos, 500)-e.gravity(0., pos)
    assert srp @ (e.sun_position_m-pos) < 0


def test_ellipsoid_and_shadow():
    e = Environment()
    b = e.binary
    p = b.states(0.)[0]
    assert_allclose(b.surface(0., p[1]+[0, 0, 58], 1), 0., atol=1e-14)
    assert e.illumination(0., p[0]+[-5000., 0., 0.]) == 0
    assert e.illumination(0., [5000., 0., 0.]) == 1


def test_jacobi_invariant():
    e = Environment(solar_tide=False, srp=False)
    y0 = np.array([3000., 0., 300., 0., -.10, 0.])
    def rhs(t, y):
        return np.r_[y[3:], e.gravity(t, y[:3])]
    sol = solve_ivp(rhs, (0., 20000.), y0, method="DOP853", rtol=2e-12,
                    atol=1e-11, max_step=100.)
    c = np.array([jacobi(t, y[:3], y[3:], e.binary) for t, y in zip(sol.t, sol.y.T)])
    assert np.max(abs(c-c[0])) < 1e-10


def test_saturation_and_propellant_budget():
    f, saturated = allocate_force([1, 1, 1], 500., 10., 480., 220., 1.)
    assert saturated
    assert_allclose(np.linalg.norm(f), 10)
    f, _ = allocate_force([1, 0, 0], 500., 10., 499.999, 220., 10.)
    assert_allclose(np.linalg.norm(f)*10/(220*9.80665), .001, atol=1e-12)
    f, _ = allocate_force([1, 0, 0], 480., 10., 480., 220., 1.)
    assert_allclose(f, [0, 0, 0])


def test_guidance_sign_and_nonbraking_reference():
    g = Guidance(coast_tgo_s=0)
    a, tau, _ = g.command(0, np.array([100., 0, 1000., 0, 0, -10.]), np.zeros(3))
    assert a[0] > 0
    assert_allclose(a[2], 0)
    g.kind = "pd"
    x = np.array([0, 0, 6100.*600, 0, 0, -6100.])
    assert_allclose(g.command(0, x, np.zeros(3))[0], np.zeros(3))


def test_fast_collision_cannot_tunnel_and_wrong_body():
    cfg = Scenario(solar_tide=False, srp=False)
    e = environment(cfg)
    for body in (0, 1):
        p, v, _ = e.binary.states(0.)
        # Both interval endpoints would be outside even for a direct hit.
        y = np.r_[p[body]+[0, 0, -3000], v[body]+[0, 0, 6100], 500.]
        t, out, event = advance(0., y, 1., np.zeros(3), cfg, e)
        assert event == body
        assert 0 < t < 1
        assert abs(e.binary.surface(t, out[:3], body)) < 1e-7
