"""Sampled closed loop with continuous collision events and separate navigation."""
from dataclasses import dataclass, asdict, field
import numpy as np
from scipy.integrate import solve_ivp
from .model import Binary, Environment, G0
from .guidance import Guidance, allocate_force
from .camera import Camera, Detection, pointing_rotation
from .navigation import BearingEKF


@dataclass
class Scenario:
    name: str = "fast_terminal"
    target: int = 1
    speed_mps: float = 6100.
    duration_s: float = 600.
    max_time_s: float = 650.
    control_dt_s: float = 1.
    mass_kg: float = 500.
    dry_mass_kg: float = 480.
    max_force_n: float = 10.
    isp_s: float = 220.
    transverse_position_m: list = field(default_factory=lambda: [1000., -500.])
    transverse_velocity_mps: list = field(default_factory=lambda: [0.5, -0.2])
    guidance: str = "zem"
    navigation: str = "truth"
    coast_tgo_s: float = 5.
    pd_frequency_radps: float = 0.012
    solar_tide: bool = True
    srp: bool = True
    phase_rad: float = 0.
    separation_m: float = 1190.
    period_s: float = 42912.
    secondary_fraction: float = 0.00925
    area_m2: float = 5.
    reflectivity: float = 1.5
    seed: int = 7
    pixel_sigma: float = 0.35
    dropout_probability: float = 0.05
    blackout_s: list = field(default_factory=lambda: [[200., 220.]])
    initial_estimate_error: list = field(default_factory=lambda: [40., -25., 2000., .08, -.05, .4])
    prior_sigma: list = field(default_factory=lambda: [80., 80., 10000., .3, .3, 2.])
    hold_after_s: float = 5.
    max_cross_sigma_m: float = 500.
    max_fractional_range_sigma: float = .1
    rtol: float = 1e-10
    atol_position_m: float = 1e-5
    max_step_s: float = 1.

    def validate(self):
        if self.target not in (0, 1) or self.navigation not in ("truth", "bearing"):
            raise ValueError("target must be 0/1; navigation must be truth/bearing")
        if self.guidance not in ("zem", "pd", "none"):
            raise ValueError("guidance must be zem/pd/none")
        for name in ("speed_mps", "duration_s", "max_time_s", "control_dt_s",
                     "mass_kg", "dry_mass_kg", "isp_s", "pixel_sigma",
                     "hold_after_s", "max_cross_sigma_m", "max_fractional_range_sigma",
                     "rtol", "atol_position_m", "max_step_s", "pd_frequency_radps"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (self.mass_kg < self.dry_mass_kg or self.max_force_n < 0
                or not np.isfinite(self.max_force_n) or self.coast_tgo_s < 0):
            raise ValueError("Invalid mass, thrust or coast time")
        if not 0 <= self.dropout_probability <= 1:
            raise ValueError("dropout_probability outside [0,1]")
        for name, n in (("transverse_position_m", 2), ("transverse_velocity_mps", 2),
                        ("initial_estimate_error", 6), ("prior_sigma", 6)):
            v = np.asarray(getattr(self, name))
            if v.shape != (n,) or not np.all(np.isfinite(v)):
                raise ValueError(f"Invalid {name}")
        if min(self.prior_sigma) <= 0:
            raise ValueError("Prior standard deviations must be positive")
        if any(len(w) != 2 or w[0] < 0 or w[1] <= w[0] for w in self.blackout_s):
            raise ValueError("Blackout intervals must be [start,end), end>start>=0")


def environment(config):
    return Environment(
        Binary(separation_m=config.separation_m, period_s=config.period_s,
               secondary_fraction=config.secondary_fraction, phase_rad=config.phase_rad),
        solar_tide=config.solar_tide, srp=config.srp,
        area_m2=config.area_m2, reflectivity=config.reflectivity)


def relative_state(t, y, env, target):
    p, v, _ = env.binary.states(t)
    return np.r_[p[target]-y[:3], v[target]-y[3:6]]


def advance(t, y, dt, force, config, env):
    """Internal step bounded before any possible pass through a body this interval."""
    p, v, _ = env.binary.states(t)
    maxstep = min(dt, config.max_step_s)
    for body in (0, 1):
        speed_bound = np.linalg.norm(y[3:6]-v[body]) + 1. + config.max_force_n/y[6]*dt
        clearance = np.linalg.norm(y[:3]-p[body])-max(env.binary.axes_m[body])
        if clearance <= speed_bound*dt:
            maxstep = min(maxstep, .25*min(env.binary.axes_m[body])/speed_bound)
    def rhs(time, state):
        return np.r_[state[3:6], env.acceleration(time, state[:3], state[6])+force/state[6],
                     -np.linalg.norm(force)/(config.isp_s*G0)]
    def primary(time, state):
        return env.binary.surface(time, state[:3], 0)
    def secondary(time, state):
        return env.binary.surface(time, state[:3], 1)
    def closest(time, state):
        x = relative_state(time, state, env, config.target)
        return x[:3] @ x[3:]
    for event in (primary, secondary):
        event.terminal, event.direction = True, -1
    closest.terminal, closest.direction = True, 1
    sol = solve_ivp(rhs, (t, t+dt), y, method="DOP853",
                    rtol=config.rtol, atol=[config.atol_position_m]*3+[1e-8]*3+[1e-9],
                    max_step=maxstep, events=[primary, secondary, closest])
    if not sol.success:
        raise RuntimeError(sol.message)
    which = next((i for i, times in enumerate(sol.t_events) if len(times)), None)
    return float(sol.t[-1]), sol.y[:, -1], which


def simulate(config=None):
    cfg = config or Scenario()
    cfg.validate()
    env = environment(cfg)
    p, v, _ = env.binary.states(0.)
    initial = np.array([*cfg.transverse_position_m, cfg.speed_mps*cfg.duration_s,
                        *cfg.transverse_velocity_mps, -cfg.speed_mps])
    y = np.r_[p[cfg.target]-initial[:3], v[cfg.target]-initial[3:], cfg.mass_kg]
    if any(env.binary.surface(0., y[:3], i) <= 0 for i in (0, 1)):
        raise ValueError("Initial spacecraft position is inside/on a body")
    law = Guidance(kind=cfg.guidance, closing_speed_mps=cfg.speed_mps,
                   reference_duration_s=cfg.duration_s, coast_tgo_s=cfg.coast_tgo_s,
                   natural_frequency_radps=cfg.pd_frequency_radps)
    rng = np.random.default_rng(cfg.seed)
    camera = Camera()
    nav = None
    if cfg.navigation == "bearing":
        nav = BearingEKF(initial + cfg.initial_estimate_error, np.diag(np.square(cfg.prior_sigma)))
    rows = []
    t, dv_used, saturation_seconds, hold_seconds = 0., 0., 0., 0.
    status, collided_body = "TIMEOUT", None
    last_force = np.zeros(3)
    last_mass = y[6]
    while t < cfg.max_time_s-1e-10:
        truth = relative_state(t, y, env, cfg.target)
        measurement_valid, cross_sigma, range_sigma = False, 0., 0.
        if nav is not None:
            nav.predict(t, last_force, last_mass, cfg.isp_s, env, cfg.target)
            # Pose comes from estimated LOS under an explicit ideal attitude servo.
            r_ci = pointing_rotation(nav.x[:3])
            uv = camera.project(truth[:3], r_ci)
            blackout = any(start <= t < end for start, end in cfg.blackout_s)
            # Suppress centroid measurements once projected object exceeds the frame.
            angular_radius = max(env.binary.axes_m[cfg.target])/max(np.linalg.norm(truth[:3]), 1.)
            unresolved_or_complete = 2*camera.fx*angular_radius < .9*camera.width
            if (camera.contains(uv) and unresolved_or_complete and not blackout
                    and rng.random() >= cfg.dropout_probability):
                det = Detection(t, uv+rng.normal(0, cfg.pixel_sigma, 2),
                                np.eye(2)*cfg.pixel_sigma**2, cfg.target)
                measurement_valid = nav.update(det, camera, r_ci, cfg.target)
            estimate = nav.x.copy()
        else:
            estimate = truth.copy()
        drift = env.relative_drift(t, estimate, y[6], cfg.target)
        command, tgo, mode = law.command(t, estimate, drift)
        if nav is not None:
            los = estimate[:3]/np.linalg.norm(estimate[:3])
            transverse = np.eye(3)-np.outer(los, los)
            b = np.hstack([np.eye(3), max(tgo, 0)*np.eye(3)])
            future_cov = b @ nav.P @ b.T
            cross_sigma = np.sqrt(max(np.linalg.eigvalsh(transverse @ future_cov @ transverse)))
            range_sigma = np.sqrt(max(los @ nav.P[:3, :3] @ los, 0))
            stale = nav.last_accepted_s is None or t-nav.last_accepted_s > cfg.hold_after_s
            uncertain = (cross_sigma > cfg.max_cross_sigma_m or
                         range_sigma/max(np.linalg.norm(estimate[:3]), 1) > cfg.max_fractional_range_sigma)
            if mode == "GUIDANCE" and (stale or uncertain):
                command, mode = np.zeros(3), "NAV_HOLD"
        dt = min(cfg.control_dt_s, cfg.max_time_s-t)
        force, saturated = allocate_force(command, y[6], cfg.max_force_n,
                                            cfg.dry_mass_kg, cfg.isp_s, dt)
        # Telemetry records the state BEFORE applying this interval's command.
        row = dict(t_s=t, range_m=float(np.linalg.norm(truth[:3])), mass_kg=float(y[6]),
                   tgo_s=float(tgo), force_n=float(np.linalg.norm(force)),
                   saturated=int(saturated), mode=mode, measurement_valid=int(measurement_valid),
                   nis=nav.last_nis if nav is not None and measurement_valid else float("nan"),
                   cross_sigma_m=float(cross_sigma), range_sigma_m=float(range_sigma))
        for prefix, vector in (("r", truth[:3]), ("v", truth[3:]), ("rhat", estimate[:3]),
                               ("vhat", estimate[3:]), ("force", force)):
            row.update({f"{prefix}_{axis}": float(value) for axis, value in zip("xyz", vector)})
        if nav is not None:
            for k in range(6):
                row[f"variance_{k}"] = float(nav.P[k, k])
        rows.append(row)
        tnew, ynext, event = advance(t, y, dt, force, cfg, env)
        used = tnew-t
        dv_used += cfg.isp_s*G0*np.log(y[6]/ynext[6])
        saturation_seconds += saturated*used
        hold_seconds += (mode == "NAV_HOLD")*used
        last_force, last_mass = force, y[6]
        t, y = tnew, ynext
        # The force budget permits only roundoff-scale fuel overshoot.
        if y[6] < cfg.dry_mass_kg-1e-7:
            raise RuntimeError("Fuel budget violated")
        y[6] = max(y[6], cfg.dry_mass_kg)
        if event is not None:
            if event in (0, 1):
                collided_body = event
                status = "TARGET_IMPACT" if event == cfg.target else "WRONG_BODY_IMPACT"
            else:
                status = "MISS"
            break
    terminal = relative_state(t, y, env, cfg.target)
    summary = dict(
        scenario=cfg.name, guidance=cfg.guidance, navigation=cfg.navigation, seed=cfg.seed,
        status=status, success=status == "TARGET_IMPACT",
        collided_body=collided_body, terminal_time_s=t,
        terminal_center_distance_m=float(np.linalg.norm(terminal[:3])),
        terminal_relative_speed_mps=float(np.linalg.norm(terminal[3:])),
        terminal_transverse_distance_m=float(np.linalg.norm(terminal[:2])),
        delta_v_mps=float(dv_used), propellant_used_kg=float(cfg.mass_kg-y[6]),
        saturation_time_s=float(saturation_seconds), nav_hold_time_s=float(hold_seconds),
        accepted_measurements=nav.accepted if nav else 0,
        rejected_measurements=nav.rejected if nav else 0,
        total_gm_m3ps2=float(sum(env.binary.mu)))
    if collided_body is not None:
        body = collided_body
        p, v, _ = env.binary.states(t)
        rot = env.binary.body_to_inertial(t, body)
        offset = y[:3]-p[body]
        b = rot.T @ offset
        normal = rot @ (b/env.binary.axes_m[body]**2)
        normal /= np.linalg.norm(normal)
        omega = env.binary.n if body == 1 else 2*np.pi/env.binary.primary_spin_period_s
        surface_v = v[body]+np.cross([0., 0., omega], offset)
        vr = y[3:6]-surface_v
        summary["surface_event_residual"] = float(env.binary.surface(t, y[:3], body))
        summary["impact_incidence_deg"] = float(np.degrees(np.arccos(
            np.clip(-vr @ normal/np.linalg.norm(vr), -1, 1))))
    summary["terminal_state_spacecraft"] = y.tolist()
    summary["terminal_state_relative"] = terminal.tolist()
    return dict(config=asdict(cfg), summary=summary, telemetry=rows)
