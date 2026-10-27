"""Prescribed circular binary, restricted spacecraft dynamics, finite surfaces."""
from dataclasses import dataclass, field
import numpy as np

AU = 149_597_870_700.0
MU_SUN = 1.32712440018e20
G0 = 9.80665


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


@dataclass
class Binary:
    separation_m: float = 1190.0
    period_s: float = 11.92 * 3600
    secondary_fraction: float = 0.00925
    phase_rad: float = 0.0
    primary_spin_period_s: float = 2.26 * 3600
    # Geometric ellipsoids only: not an ellipsoid gravity solution.
    axes_m: np.ndarray = field(default_factory=lambda: np.array(
        [[425., 424., 310.], [88., 87., 58.]]))

    def __post_init__(self):
        self.axes_m = np.asarray(self.axes_m, dtype=float)
        if (self.separation_m <= 0 or self.period_s <= 0
                or self.primary_spin_period_s <= 0
                or not 0 < self.secondary_fraction < 1
                or self.axes_m.shape != (2, 3)
                or not np.all(np.isfinite(self.axes_m)) or np.any(self.axes_m <= 0)):
            raise ValueError("Invalid binary parameters")
        if np.max(self.axes_m[0]) + np.max(self.axes_m[1]) >= self.separation_m:
            raise ValueError("Bounding spheres of binary bodies must be separated")

    @property
    def n(self):
        return 2 * np.pi / self.period_s

    @property
    def mu(self):
        """Derive GM, rather than over-specify inconsistent mass, period, radius."""
        total = self.n**2 * self.separation_m**3
        return total * np.array([1 - self.secondary_fraction, self.secondary_fraction])

    def states(self, t):
        theta = self.phase_rad + self.n * t
        d = self.separation_m * np.array([np.cos(theta), np.sin(theta), 0.])
        dv = self.n * np.array([-d[1], d[0], 0.])
        weights = np.array([-self.secondary_fraction, 1 - self.secondary_fraction])
        p = weights[:, None] * d
        v = weights[:, None] * dv
        a = -self.n**2 * p
        return p, v, a

    def body_to_inertial(self, t, body):
        rate = 2 * np.pi / self.primary_spin_period_s if body == 0 else self.n
        return rz(self.phase_rad + rate * t)

    def surface(self, t, spacecraft_position, body):
        """Zero on surface, positive outside, negative inside."""
        p = self.states(t)[0][body]
        b = self.body_to_inertial(t, body).T @ (spacecraft_position - p)
        return np.sum((b / self.axes_m[body])**2) - 1


@dataclass
class Environment:
    binary: Binary = field(default_factory=Binary)
    solar_tide: bool = True
    srp: bool = True
    sun_position_m: np.ndarray = field(default_factory=lambda: np.array([1.5 * AU, 0., 0.]))
    area_m2: float = 5.0
    reflectivity: float = 1.5  # Cr, not (1+Cr)
    pressure_at_au_pa: float = 4.56e-6

    def __post_init__(self):
        self.sun_position_m = np.asarray(self.sun_position_m, dtype=float)
        if (self.sun_position_m.shape != (3,) or not np.all(np.isfinite(self.sun_position_m))
                or np.linalg.norm(self.sun_position_m) <= 0
                or self.area_m2 < 0 or not 0 <= self.reflectivity <= 2):
            raise ValueError("Invalid Sun vector, area or reflectivity")

    def gravity(self, t, position):
        d = np.asarray(position) - self.binary.states(t)[0]
        radii = np.linalg.norm(d, axis=1)
        if np.any(radii == 0):
            raise ValueError("Point-mass singularity; stop propagation at the surface")
        return -np.sum(self.binary.mu[:, None] * d / radii[:, None]**3, axis=0)

    def gravity_gradient(self, t, position):
        d = np.asarray(position) - self.binary.states(t)[0]
        out = np.zeros((3, 3))
        for mu, v in zip(self.binary.mu, d):
            r = np.linalg.norm(v)
            out += mu * (3 * np.outer(v, v) / r**5 - np.eye(3) / r**3)
        return out

    def illumination(self, t, position):
        """Binary hard umbra: ray from spacecraft to Sun intersects an ellipsoid."""
        direction = self.sun_position_m - position
        distance = np.linalg.norm(direction)
        direction /= distance
        centers = self.binary.states(t)[0]
        for body in (0, 1):
            rot = self.binary.body_to_inertial(t, body).T
            q = rot @ (position - centers[body]) / self.binary.axes_m[body]
            d = rot @ direction / self.binary.axes_m[body]
            aa, bb, cc = d @ d, 2 * q @ d, q @ q - 1
            disc = bb * bb - 4 * aa * cc
            if disc >= 0:
                near = (-bb - np.sqrt(disc)) / (2 * aa)
                far = (-bb + np.sqrt(disc)) / (2 * aa)
                if far > 0 and near < distance:
                    return 0.0
        return 1.0

    def acceleration(self, t, position, mass):
        a = self.gravity(t, position)
        s = self.sun_position_m
        d = s - position
        if self.solar_tide:
            # Indirect term subtracts acceleration of the frame origin.
            # Stable algebra avoids subtracting two almost equal solar forces.
            q = (position @ position - 2 * s @ position) / (s @ s)
            f = np.expm1(-1.5 * np.log1p(q))
            a += MU_SUN / np.linalg.norm(s)**3 * (s * f - position * (1 + f))
        if self.srp:
            a -= (self.illumination(t, position) * self.pressure_at_au_pa
                  * self.reflectivity * self.area_m2 / mass
                  * (AU / np.linalg.norm(d))**2 * d / np.linalg.norm(d))
        return a

    def relative_drift(self, t, relative_state, mass, target=1):
        """r = r_target-r_spacecraft; v = v_target-v_spacecraft."""
        p, _, a = self.binary.states(t)
        return a[target] - self.acceleration(t, p[target] - relative_state[:3], mass)

    def relative_jacobian(self, t, relative_state, mass, target=1):
        """Central differences include solar/SRP terms; nonsmooth at shadow edges."""
        jac = np.zeros((6, 6))
        jac[:3, 3:] = np.eye(3)
        h = 0.1
        for k in range(3):
            dx = np.zeros(6)
            dx[k] = h
            jac[3:, k] = (self.relative_drift(t, relative_state + dx, mass, target)
                         - self.relative_drift(t, relative_state - dx, mass, target)) / (2*h)
        return jac


def inertial_to_rotating(t, position, velocity, binary):
    rot = rz(binary.phase_rad + binary.n*t).T
    r = rot @ position
    v = rot @ velocity - np.cross([0, 0, binary.n], r)
    return r, v


def jacobi(t, position, velocity, binary):
    """Dimensioned C, conserved ONLY for point binary with no thrust/Sun/SRP."""
    r, v = inertial_to_rotating(t, position, velocity, binary)
    centers = binary.states(t)[0]
    return (binary.n**2 * (r[0]**2 + r[1]**2)
            + 2 * np.sum(binary.mu / np.linalg.norm(position - centers, axis=1))
            - v @ v)
