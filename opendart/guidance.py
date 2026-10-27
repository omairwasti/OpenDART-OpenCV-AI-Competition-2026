"""Feedback produces spacecraft acceleration in inertially fixed axes."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Guidance:
    kind: str = "zem"
    closing_speed_mps: float = 6100.0
    reference_duration_s: float = 600.0
    minimum_tgo_s: float = 2.0
    coast_tgo_s: float = 5.0
    natural_frequency_radps: float = 0.012
    damping_ratio: float = 1.0

    def command(self, t, estimate, drift):
        r, v = estimate[:3], estimate[3:]
        vv = v @ v
        tau = -(r @ v) / vv if vv > 1e-16 else -1.
        if self.kind == "none":
            return np.zeros(3), tau, "BALLISTIC"
        if tau <= 0:
            return np.zeros(3), tau, "OPENING"
        if tau <= self.coast_tgo_s:
            return np.zeros(3), tau, "TERMINAL_COAST"
        if self.kind == "zem":
            # Minimum integrated squared acceleration, free terminal velocity,
            # locally constant natural relative acceleration, frozen time-to-go.
            h = max(tau, self.minimum_tgo_s)
            zem = r + v*h + 0.5*drift*h*h
            return 3*zem/(h*h), tau, "GUIDANCE"
        if self.kind == "pd":
            # Track a closing trajectory, not r=0,v=0 (which would rendezvous).
            e = np.array([0., 0., 1.])
            rref = self.closing_speed_mps * (self.reference_duration_s - t) * e
            vref = -self.closing_speed_mps * e
            w, zeta = self.natural_frequency_radps, self.damping_ratio
            return drift + w*w*(r-rref) + 2*zeta*w*(v-vref), tau, "GUIDANCE"
        raise ValueError("guidance must be zem, pd or none")


def allocate_force(acceleration, mass, max_force_n, dry_mass_kg, isp_s, dt):
    """Ideal omnidirectional thrust; norm bound and per-interval propellant budget."""
    requested = mass * np.asarray(acceleration, dtype=float)
    limit = min(max_force_n, max(mass-dry_mass_kg, 0.) * isp_s * 9.80665 / dt)
    norm = np.linalg.norm(requested)
    force = requested * min(1., limit/norm) if norm > 0 else requested
    return force, bool(norm > limit + 1e-12)
