"""Physical XY terminal approach driven by detected pixels, not state truth.

The image is a deliberately simplified pinhole disk renderer. The independent
planar bearing is horizontal pixel u. Optional apparent-size updates require a
known physical radius; this is not monocular range recovered without a prior.
"""
from dataclasses import dataclass, field, asdict
import math
from pathlib import Path
import numpy as np
from .model import Binary, Environment
from .guidance import allocate_force
from .simulation import Scenario, advance
from .vision import AsteroidDetector, AsteroidTracker, require_opencv


@dataclass
class PlanarConfig:
    name: str = "pixel_closed_loop_2d"
    period_s: float = 42900.0
    spin_period_s: float = 8136.0
    separation_m: float = 1190.0
    phase_rad: float = 0.0
    epoch_s: float = 42900.0
    duration_s: float = 600.0
    max_time_s: float = 650.0
    dt_s: float = 1.0
    initial_range_m: float = 12000.0
    initial_cross_m: float = 500.0
    speed_mps: float = 20.0
    cross_velocity_mps: float = .2
    mass_kg: float = 500.0
    dry_mass_kg: float = 480.0
    thrust_n: float = 10.0
    isp_s: float = 220.0
    coast_tgo_s: float = 12.0
    control: bool = True
    measurements: bool = True
    use_apparent_size: bool = True
    camera_f_px: float = 700.0
    camera_width: int = 512
    camera_height: int = 512
    camera_rate_radps: float = .04
    noise_dn: float = 1.0
    dropout_probability: float = .02
    blackout_s: list = field(default_factory=lambda: [[200., 220.]])
    seed: int = 7
    estimate_error: list = field(default_factory=lambda: [60., -250., .1, -.3])
    prior_sigma: list = field(default_factory=lambda: [100., 800., .5, 1.5])
    acceleration_psd: float = 1e-7
    size_fractional_sigma: float = .10
    # Pooled offline synthetic calibration, rounded training median (see visual guide).
    # Predict detector extent, not an unbiased physical limb. Zero restores v0.2.
    size_radius_bias_px: float = 1.07
    # Navigation prior only. Renderer/contact geometry remain unchanged.
    navigation_radius_m: float = 88.0
    hold_after_s: float = 5.0
    maximum_fractional_range_sigma: float = .5
    require_opencv5: bool = True
    fps: float = 24.0
    overview_frames: int = 192
    reference_image: str | None = None

    def validate(self):
        positive = ("period_s", "spin_period_s", "separation_m", "duration_s",
                    "max_time_s", "dt_s", "initial_range_m", "speed_mps", "mass_kg",
                    "dry_mass_kg", "isp_s", "camera_f_px", "camera_rate_radps",
                    "size_fractional_sigma", "hold_after_s", "fps",
                    "maximum_fractional_range_sigma", "navigation_radius_m")
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        nonnegative = ("thrust_n", "noise_dn", "acceleration_psd", "coast_tgo_s",
                       "size_radius_bias_px")
        for name in nonnegative:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        for name in ("initial_cross_m", "cross_velocity_mps", "phase_rad", "epoch_s"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.mass_kg < self.dry_mass_kg:
            raise ValueError("Invalid mass/force")
        if not np.isfinite(self.dropout_probability) or not 0 <= self.dropout_probability <= 1:
            raise ValueError("Invalid image degradation")
        for name, minimum in (("camera_width", 64), ("camera_height", 64),
                              ("overview_frames", 0), ("seed", 0)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("estimate_error", "prior_sigma"):
            vector = np.asarray(getattr(self, name), dtype=float)
            if vector.shape != (4,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} requires four finite values")
            if name == "prior_sigma" and np.any(vector <= 0):
                raise ValueError("Prior standard deviations must be positive")
        for window in self.blackout_s:
            values = np.asarray(window, dtype=float)
            if values.shape != (2,) or not np.all(np.isfinite(values)) or not 0 <= values[0] < values[1]:
                raise ValueError("Invalid blackout interval")
        for name in ("control", "measurements", "use_apparent_size", "require_opencv5"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        # duration_s is retained as historical metadata; range and max_time_s
        # define this experiment. Do not change its termination semantics.


def camera_axes(theta):
    """Forward along theta; image right is clockwise from forward in world XY."""
    return np.array([np.sin(theta), -np.cos(theta)]), np.array([np.cos(theta), np.sin(theta)])


def measurement_model(x, theta, focal_px, radius_m, cx, use_size=True, radius_bias_px=0.):
    """Predict centroid and log detector extent; additive bias is pixel-domain."""
    right, forward = camera_axes(theta)
    xcam, zcam = right @ x[:2], forward @ x[:2]
    if not np.all(np.isfinite([xcam, zcam])) or zcam <= 0:
        return None, None
    u = focal_px*xcam/zcam+cx
    h = np.zeros((2 if use_size else 1, 4))
    h[0, :2] = focal_px*(right/zcam-xcam*forward/zcam**2)
    if use_size:
        ideal_radius = focal_px*radius_m/zcam
        observed_radius = ideal_radius+radius_bias_px
        h[1, :2] = -forward/zcam*(ideal_radius/observed_radius)
        return np.array([u, np.log(observed_radius)]), h
    return np.array([u]), h


UPDATE_STATUSES = ("ACCEPTED", "NO_CANDIDATE", "ASSOCIATION_FAILED", "CLIPPED",
                   "TOO_SMALL", "INVALID_GEOMETRY", "INVALID_MEASUREMENT", "NIS_REJECTED")
MISSING_STATUSES = ("NO_CANDIDATE", "ASSOCIATION_FAILED")


def range_sigma(x, covariance):
    """Linearized radial sigma from the estimate alone; not a coverage claim."""
    distance = np.linalg.norm(x[:2])
    los = x[:2]/max(distance, 1.)
    return math.sqrt(max(float(los@covariance[:2, :2]@los), 0.))


class PlanarEKF:
    def __init__(self, x, covariance, cfg, env):
        self.x = np.asarray(x, float).copy()
        self.P = np.asarray(covariance, float).copy()
        self.cfg, self.env = cfg, env
        self.accepted, self.rejected = 0, 0
        self.last_update_s = None
        self.nis = float("nan")
        self.update_counts = dict.fromkeys(UPDATE_STATUSES, 0)
        self.last_diagnostics = {}

    def drift(self, elapsed, x, mass):
        embedded = np.r_[x[:2], 0., x[2:], 0.]
        return self.env.relative_drift(self.cfg.epoch_s+elapsed, embedded, mass, 1)[:2]

    def predict(self, elapsed, dt, force_xy, mass_start):
        force = np.asarray(force_xy, float)
        flow = np.linalg.norm(force)/(self.cfg.isp_s*9.80665)
        def derivative(s, x):
            mass = mass_start-flow*s
            return np.r_[x[2:], self.drift(elapsed+s, x, mass)-force/mass]
        # Fine RK4 is inexpensive for four states; covariance is linearized locally.
        steps = max(1, int(np.ceil(dt/.5)))
        hstep = dt/steps
        a = np.zeros((4, 4)); a[:2, 2:] = np.eye(2)
        target = self.env.binary.states(self.cfg.epoch_s+elapsed)[0][1]
        p = target-np.r_[self.x[:2], 0.]
        a[2:, :2] = self.env.gravity_gradient(self.cfg.epoch_s+elapsed, p)[:2, :2]
        for k in range(steps):
            s = k*hstep
            k1 = derivative(s, self.x)
            k2 = derivative(s+hstep/2, self.x+hstep*k1/2)
            k3 = derivative(s+hstep/2, self.x+hstep*k2/2)
            k4 = derivative(s+hstep, self.x+hstep*k3)
            self.x += hstep*(k1+2*k2+2*k3+k4)/6
        phi = np.eye(4)+dt*a+.5*dt*dt*(a@a)
        q = self.cfg.acceleration_psd*np.block([
            [dt**3/3*np.eye(2), dt**2/2*np.eye(2)],
            [dt**2/2*np.eye(2), dt*np.eye(2)]])
        self.P = phi@self.P@phi.T+q
        self.P = (self.P+self.P.T)/2

    def update(self, candidate, theta, elapsed, candidate_count=0):
        """One classified attempt per frame. Missing data are not NIS rejections.

        This remains a synchronous Candidate interface; elapsed is the caller's
        measurement epoch. No truth, latency reconstruction or identity is added.
        """
        cfg = self.cfg
        self.nis = float("nan")
        diagnostics = dict(measurement_time_s=float(elapsed), update_status="",
                           rejection_reason="", measurement_rejected=0,
                           predicted_u_px=float("nan"), predicted_radius_px=float("nan"),
                           innovation_u_px=float("nan"), innovation_log_radius=float("nan"),
                           innovation_var_u_px2=float("nan"), innovation_var_log_radius=float("nan"),
                           innovation_cov_u_log_radius=float("nan"),
                           measurement_u_sigma_px=float("nan"), measurement_log_radius_sigma=float("nan"))
        self.last_diagnostics = diagnostics

        def finish(status):
            self.update_counts[status] += 1
            diagnostics["update_status"] = status
            diagnostics["rejection_reason"] = "" if status == "ACCEPTED" else status
            rejected = status != "ACCEPTED" and status not in MISSING_STATUSES
            diagnostics["measurement_rejected"] = int(rejected)
            self.rejected += int(rejected)
            return status == "ACCEPTED"

        full_prediction, full_h = measurement_model(
            self.x, theta, cfg.camera_f_px, cfg.navigation_radius_m, (cfg.camera_width-1)/2,
            True, cfg.size_radius_bias_px)
        if full_prediction is not None:
            diagnostics["predicted_u_px"] = float(full_prediction[0])
            diagnostics["predicted_radius_px"] = float(np.exp(full_prediction[1]))
        if candidate is None:
            return finish("ASSOCIATION_FAILED" if candidate_count else "NO_CANDIDATE")
        if candidate.touches_border:
            return finish("CLIPPED")
        uv, covariance = np.asarray(candidate.centroid_px), np.asarray(candidate.covariance_px2)
        if (uv.shape != (2,) or covariance.shape != (2, 2)
                or not np.all(np.isfinite(uv)) or not np.all(np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T)
                or np.min(np.linalg.eigvalsh(covariance)) <= 0
                or not np.isfinite(candidate.radius_px) or candidate.radius_px <= 0):
            return finish("INVALID_MEASUREMENT")
        if candidate.radius_px < 2.:
            return finish("TOO_SMALL")
        if full_prediction is None:
            return finish("INVALID_GEOMETRY")
        predicted = full_prediction if cfg.use_apparent_size else full_prediction[:1]
        h = full_h if cfg.use_apparent_size else full_h[:1]
        z = [candidate.centroid_px[0]]
        variances = [candidate.covariance_px2[0, 0]]
        diagnostics["measurement_u_sigma_px"] = math.sqrt(variances[0])
        if cfg.use_apparent_size:
            z.append(np.log(candidate.radius_px))
            variances.append(max(.5/candidate.radius_px, cfg.size_fractional_sigma)**2)
            diagnostics["measurement_log_radius_sigma"] = math.sqrt(variances[1])
        r = np.diag(variances)
        residual = np.asarray(z)-predicted
        diagnostics["innovation_u_px"] = float(residual[0])
        if cfg.use_apparent_size:
            diagnostics["innovation_log_radius"] = float(residual[1])
        s = h@self.P@h.T+r
        diagnostics["innovation_var_u_px2"] = float(s[0, 0])
        if cfg.use_apparent_size:
            diagnostics["innovation_var_log_radius"] = float(s[1, 1])
            diagnostics["innovation_cov_u_log_radius"] = float(s[0, 1])
        self.nis = float(residual @ np.linalg.solve(s, residual))
        if self.nis > (9.21034 if cfg.use_apparent_size else 6.63490):
            return finish("NIS_REJECTED")
        gain = np.linalg.solve(s, h@self.P).T
        self.x += gain@residual
        ikh = np.eye(4)-gain@h
        self.P = ikh@self.P@ikh.T+gain@r@gain.T
        self.P = (self.P+self.P.T)/2
        self.accepted += 1
        self.last_update_s = elapsed
        return finish("ACCEPTED")


class CameraRenderer:
    """Forward renderer only. Detector receives its image, never projected truth."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.stars = np.random.default_rng(cfg.seed+100).uniform(
            [0, 0], [cfg.camera_width, cfg.camera_height], (35, 2))
        self.templates = None
        if cfg.reference_image is not None:
            cv = require_opencv()
            source = cv.imread(str(Path(cfg.reference_image).expanduser()), cv.IMREAD_GRAYSCALE)
            if source is None:
                raise ValueError(f"Cannot read reference appearance image: {cfg.reference_image}")
            objects = AsteroidDetector().detect(source)
            secondary = AsteroidTracker().update(objects, 0.)
            if len(objects) < 2 or secondary is None:
                raise ValueError("Reference image must resolve a primary and secondary hypothesis")
            self.templates = []
            for candidate in (objects[0], secondary):
                u, v = candidate.centroid_px
                _, _, w, h = candidate.bbox_xywh
                side = max(w, h)+4
                patch = cv.getRectSubPix(source, (side, side), (float(u), float(v)))
                self.templates.append(patch.astype(np.float32)/255.)

    def render(self, time_s, spacecraft_xy, theta, binary, blank=False):
        cv = require_opencv()
        cfg = self.cfg
        image = np.full((cfg.camera_height, cfg.camera_width), 3., np.float32)
        # Star points remain intentionally weak and below the resolved-blob cutoff.
        for u, v in self.stars:
            cv.circle(image, (int(u), int(v)), 0, 22., -1)
        if not blank:
            right, forward = camera_axes(theta)
            positions = binary.states(time_s)[0][:, :2]
            projected = []
            for body, radius_m in enumerate([425., 88.]):
                delta = positions[body]-spacecraft_xy
                z = forward@delta
                if z <= radius_m:
                    continue
                u = cfg.camera_f_px*(right@delta)/z+(cfg.camera_width-1)/2
                r = cfg.camera_f_px*radius_m/z
                projected.append((z, body, float(u), float(r)))
            # Far disks first: nearer projected disks can occult them.
            for z, body, u, r in sorted(projected, reverse=True):
                if r < .6 or u+r < 0 or u-r >= cfg.camera_width:
                    continue
                cy = (cfg.camera_height-1)/2
                x0, x1 = max(0, int(u-r-2)), min(cfg.camera_width, int(u+r+3))
                y0, y1 = max(0, int(cy-r-2)), min(cfg.camera_height, int(cy+r+3))
                yy, xx = np.mgrid[y0:y1, x0:x1]
                xn, yn = (xx-u)/r, (yy-cy)/r
                radial = xn*xn+yn*yn
                inside = radial <= 1
                spin = (2*np.pi/cfg.spin_period_s if body == 0 else binary.n)*time_s
                texture = 1+.07*np.sin(14*xn*np.cos(spin)+10*yn*np.sin(spin)+body)
                intensity = (155+35*np.sqrt(np.maximum(1-radial, 0)))*texture
                if self.templates is not None:
                    # Reference optical texture only. A visible disk pedestal
                    # retains the declared geometric silhouette. This is not
                    # a calibrated, illuminated shape reconstruction.
                    template = self.templates[body]
                    spin_delta = (2*np.pi/cfg.spin_period_s if body == 0 else binary.n)*(time_s-cfg.epoch_s)
                    a, b = np.cos(spin_delta), np.sin(spin_delta)
                    tx, ty = a*xn+b*yn, -b*xn+a*yn
                    mx = ((tx+1)*.5*(template.shape[1]-1)).astype(np.float32)
                    my = ((ty+1)*.5*(template.shape[0]-1)).astype(np.float32)
                    appearance = cv.remap(template, mx, my, cv.INTER_LINEAR,
                                          borderMode=cv.BORDER_CONSTANT, borderValue=0.)
                    intensity = 70.+125.*appearance
                roi = image[y0:y1, x0:x1]
                roi[inside] = intensity[inside]
        image = cv.GaussianBlur(image, (3, 3), .55)
        image += self.rng.normal(0, cfg.noise_dn, image.shape).astype(np.float32)
        return np.clip(image, 0, 255).astype(np.uint8)


def simulate_planar(cfg=None, frame_callback=None):
    cfg = cfg or PlanarConfig()
    cfg.validate()
    cv = require_opencv()
    if cfg.require_opencv5 and int(cv.__version__.split(".")[0]) != 5:
        raise RuntimeError(f"OpenCV 5 required for this run; loaded {cv.__version__}. "
                           "Install the vision extra or set require_opencv5=false explicitly for compatibility testing.")
    binary = Binary(period_s=cfg.period_s, separation_m=cfg.separation_m,
                    phase_rad=cfg.phase_rad, primary_spin_period_s=cfg.spin_period_s)
    # This planar educational mode isolates binary gravity and control.
    env = Environment(binary, solar_tide=False, srp=False)
    plant_cfg = Scenario(target=1, control_dt_s=cfg.dt_s, mass_kg=cfg.mass_kg,
                         dry_mass_kg=cfg.dry_mass_kg, max_force_n=cfg.thrust_n,
                         isp_s=cfg.isp_s, max_step_s=.5)
    p, v, _ = binary.states(cfg.epoch_s)
    initial = np.array([cfg.initial_cross_m, cfg.initial_range_m,
                        cfg.cross_velocity_mps, -cfg.speed_mps])
    y = np.r_[p[1]-np.r_[initial[:2], 0.], v[1]-np.r_[initial[2:], 0.], cfg.mass_kg]
    nav = PlanarEKF(initial+cfg.estimate_error, np.diag(np.square(cfg.prior_sigma)), cfg, env)
    detector = AsteroidDetector()
    tracker = AsteroidTracker()
    renderer = CameraRenderer(cfg)
    rng = np.random.default_rng(cfg.seed+1)
    theta = math.atan2(nav.x[1], nav.x[0])
    rows, elapsed = [], 0.
    total_dv, saturated_s, held_s = 0., 0., 0.
    status, event = "TIMEOUT", None
    last_image = last_frame = None
    while elapsed < cfg.max_time_s-1e-10:
        time_s = cfg.epoch_s+elapsed
        p, v, _ = binary.states(time_s)
        truth = np.r_[(p[1]-y[:3])[:2], (v[1]-y[3:6])[:2]]
        desired = math.atan2(nav.x[1], nav.x[0])
        delta_angle = math.atan2(math.sin(desired-theta), math.cos(desired-theta))
        theta += np.clip(delta_angle, -cfg.camera_rate_radps*cfg.dt_s,
                         cfg.camera_rate_radps*cfg.dt_s)
        blank = (not cfg.measurements or any(a <= elapsed < b for a, b in cfg.blackout_s)
                 or rng.random() < cfg.dropout_probability)
        image = renderer.render(time_s, y[:2], theta, binary, blank)
        candidates = detector.detect(image)
        pred, h = measurement_model(nav.x, theta, cfg.camera_f_px, cfg.navigation_radius_m,
                                     (cfg.camera_width-1)/2, cfg.use_apparent_size, cfg.size_radius_bias_px)
        prior_uv = None if pred is None else np.array([pred[0], (cfg.camera_height-1)/2])
        pcov = np.eye(2)*9.
        if pred is not None:
            pcov[0, 0] = max(float(h[0]@nav.P@h[0]), .1)
        selected = tracker.update(candidates, elapsed, prior_uv, pcov)
        accepted = nav.update(selected, theta, elapsed, candidate_count=len(candidates))
        r, relv = nav.x[:2], nav.x[2:]
        vv = float(relv@relv)
        tau = -float(r@relv)/vv if vv > 1e-12 else -1.
        requested = np.zeros(2)
        mode = "GUIDANCE"
        stale = nav.last_update_s is None or elapsed-nav.last_update_s > cfg.hold_after_s
        sigma_range = range_sigma(nav.x, nav.P)
        frac_sigma = sigma_range/max(np.linalg.norm(r), 1.)
        if not cfg.control:
            mode = "BALLISTIC"
        elif tau <= 0:
            mode = "OPENING"
        elif tau <= cfg.coast_tgo_s:
            mode = "TERMINAL_COAST"
        elif stale or frac_sigma > cfg.maximum_fractional_range_sigma:
            mode = "NAV_HOLD"
        else:
            drift = nav.drift(elapsed, nav.x, y[6])
            requested = 3*(r+tau*relv+.5*tau*tau*drift)/max(tau*tau, 4.)
        dt = min(cfg.dt_s, cfg.max_time_s-elapsed)
        force, saturated = allocate_force(np.r_[requested, 0.], y[6], cfg.thrust_n,
                                           cfg.dry_mass_kg, cfg.isp_s, dt)
        row = dict(elapsed_s=elapsed, physical_time_s=time_s, mode=mode, tracker=tracker.status,
                   accepted=int(accepted), candidates=len(candidates), tgo_s=tau,
                   spacecraft_x_m=float(y[0]), spacecraft_y_m=float(y[1]),
                   primary_x_m=float(p[0,0]), primary_y_m=float(p[0,1]),
                   target_x_m=float(p[1,0]), target_y_m=float(p[1,1]),
                   relative_x_m=float(truth[0]), relative_y_m=float(truth[1]),
                   estimate_x_m=float(nav.x[0]), estimate_y_m=float(nav.x[1]),
                   position_error_m=float(np.linalg.norm(nav.x[:2]-truth[:2])),
                   true_range_m=float(np.linalg.norm(truth[:2])),
                   estimated_range_m=float(np.linalg.norm(nav.x[:2])),
                   force_x_n=float(force[0]), force_y_n=float(force[1]),
                   force_n=float(np.linalg.norm(force)), mass_kg=float(y[6]),
                   theta_rad=float(theta), fractional_range_sigma=frac_sigma,
                   nis=nav.nis,
                   measured_u_px=float(selected.centroid_px[0]) if selected else float("nan"),
                   measured_radius_px=float(selected.radius_px) if selected else float("nan"))
        # Measurement predictions/innovations refer to the PRIOR; state/P below
        # refer to the POST-update estimate. Truth is evaluation only.
        row.update(nav.last_diagnostics)
        row.update(selected_score=float(selected.score) if selected else float("nan"),
                   measured_v_px=float(selected.centroid_px[1]) if selected else float("nan"),
                   estimate_vx_mps=float(nav.x[2]), estimate_vy_mps=float(nav.x[3]),
                   estimate_range_sigma_m=sigma_range,
                   estimate_position_sigma_m=math.sqrt(max(float(np.trace(nav.P[:2, :2])), 0.)),
                   nav_stale=int(stale),
                   seconds_since_accepted=elapsed-nav.last_update_s if nav.last_update_s is not None else float("nan"),
                   nav_hold_reason=("STALE" if stale else "RANGE_UNCERTAINTY") if mode == "NAV_HOLD" else "",
                   eval_true_range_m=row["true_range_m"], eval_position_error_m=row["position_error_m"],
                   eval_relative_x_m=float(truth[0]), eval_relative_y_m=float(truth[1]),
                   eval_relative_vx_mps=float(truth[2]), eval_relative_vy_mps=float(truth[3]),
                   eval_range_error_m=row["estimated_range_m"]-row["true_range_m"],
                   eval_range_error_sigma=(row["estimated_range_m"]-row["true_range_m"])/sigma_range
                                          if sigma_range > 0 else float("nan"))
        for i in range(4):
            for j in range(i, 4):
                row[f"estimate_cov_{i}{j}"] = float(nav.P[i, j])
        rows.append(row)
        if frame_callback is not None:
            last_frame = frame_callback(cfg, binary, row, image, candidates, selected, rows)
        last_image = image
        absolute_next, ynext, event = advance(time_s, y, dt, force, plant_cfg, env)
        used = absolute_next-time_s
        total_dv += cfg.isp_s*9.80665*np.log(y[6]/ynext[6])
        saturated_s += saturated*used
        held_s += (mode == "NAV_HOLD")*used
        nav.predict(elapsed, used, force[:2], y[6])
        elapsed += used
        y = ynext
        if event is not None:
            status = ("TARGET_IMPACT" if event == 1 else
                      "WRONG_BODY_IMPACT" if event == 0 else "MISS")
            break
    p, v, _ = binary.states(cfg.epoch_s+elapsed)
    terminal = np.r_[p[1]-y[:3], v[1]-y[3:6]]
    final_range = float(np.linalg.norm(nav.x[:2]))
    true_final_range = float(np.linalg.norm(terminal[:2]))
    final_sigma = range_sigma(nav.x, nav.P)
    summary = dict(name=cfg.name, status=status, success=status == "TARGET_IMPACT",
                   dimensionality="physical XY plane, perspective disk camera",
                   observation="OpenCV-detected centroid and known-radius apparent size" if cfg.use_apparent_size
                               else "OpenCV-detected horizontal bearing only",
                   appearance="DRACO patches on modeled visible disks" if cfg.reference_image else "analytic textured disks",
                   opencv_version=cv.__version__, period_s=cfg.period_s,
                   spin_period_s=cfg.spin_period_s, elapsed_s=float(elapsed),
                   terminal_center_distance_m=float(np.linalg.norm(terminal[:3])),
                   terminal_relative_speed_mps=float(np.linalg.norm(terminal[3:])),
                   delta_v_mps=float(total_dv), propellant_used_kg=float(cfg.mass_kg-y[6]),
                   accepted_updates=nav.accepted, rejected_updates=nav.rejected,
                   attempted_updates=sum(nav.update_counts.values()),
                   missing_updates=sum(nav.update_counts[k] for k in MISSING_STATUSES),
                   nis_rejected_updates=nav.update_counts["NIS_REJECTED"],
                   update_reason_counts=nav.update_counts,
                   telemetry_schema_version=2,
                   rejected_updates_definition="Selected measurements rejected; excludes missing/association failures. See update_reason_counts.",
                   size_radius_bias_px=cfg.size_radius_bias_px,
                   nav_hold_s=float(held_s), saturation_s=float(saturated_s),
                   final_estimation_error_m=float(np.linalg.norm(nav.x[:2]-terminal[:2])),
                   eval_peak_position_error_m=max(r["eval_position_error_m"] for r in rows),
                   eval_final_true_range_m=true_final_range, estimate_final_range_m=final_range,
                   estimate_final_range_sigma_m=final_sigma,
                   eval_final_range_error_sigma=(final_range-true_final_range)/final_sigma if final_sigma > 0 else None,
                   eval_max_abs_sampled_range_error_sigma=max(
                       (abs(r["eval_range_error_sigma"]) for r in rows if np.isfinite(r["eval_range_error_sigma"])),
                       default=None),
                   orbital_advance_deg=float(360*elapsed/cfg.period_s),
                   primary_spin_advance_deg=float(360*elapsed/cfg.spin_period_s),
                   last_observation_elapsed_s=float(rows[-1]["elapsed_s"]),
                   terminal_state_spacecraft=y.tolist())
    return dict(config=asdict(cfg), summary=summary, telemetry=rows,
                final_camera=last_image, final_dashboard=last_frame)
