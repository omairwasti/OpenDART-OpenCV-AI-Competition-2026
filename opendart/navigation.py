"""Optional synthetic-bearing EKF: assumes known binary ephemeris and attitude."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


class BearingEKF:
    def __init__(self, x, covariance, time_s=0., acceleration_psd=1e-8):
        self.x = np.asarray(x, dtype=float).copy()
        self.P = np.asarray(covariance, dtype=float).copy()
        self.time_s = float(time_s)
        self.acceleration_psd = acceleration_psd  # m²/s³, continuous white acceleration
        self.accepted = 0
        self.rejected = 0
        self.last_nis = float("nan")
        self.last_accepted_s = None

    def predict(self, time_s, force, mass, isp_s, env, target):
        dt = time_s - self.time_s
        if dt < -1e-9:
            raise ValueError("Cannot predict backwards")
        if dt <= 1e-12:
            return
        flow = np.linalg.norm(force)/(isp_s*9.80665)
        t0 = self.time_s
        def rhs(t, x):
            m = mass - flow*(t-t0)
            return np.r_[x[3:], env.relative_drift(t, x, m, target) - force/m]
        a = env.relative_jacobian(t0, self.x, mass, target)
        # Van Loan integral for locally frozen linearization.
        q = np.zeros((6, 6))
        q[3:, 3:] = np.eye(3)*self.acceleration_psd
        block = np.block([[a, q], [np.zeros((6, 6)), -a.T]]) * dt
        exponential = expm(block)
        phi = exponential[:6, :6]
        qd = exponential[:6, 6:] @ phi.T
        sol = solve_ivp(rhs, (t0, time_s), self.x, rtol=1e-9,
                        atol=np.array([1e-5]*3 + [1e-8]*3), max_step=min(dt, 1.))
        if not sol.success:
            raise RuntimeError(sol.message)
        self.x = sol.y[:, -1]
        self.P = phi @ self.P @ phi.T + qd
        self.P = (self.P+self.P.T)/2
        self.time_s = time_s

    def update(self, detection, camera, R_CI, target, gate=9.21034):
        """2-DOF 99% chi-square gate; rejects stale data instead of applying at now."""
        if (not detection.valid or detection.target_id != target
                or abs(detection.timestamp_s-self.time_s) > 1e-6):
            self.rejected += 1
            return False
        z = np.asarray(detection.uv_px, dtype=float)
        r = np.asarray(detection.covariance_px2, dtype=float)
        if (z.shape != (2,) or r.shape != (2, 2)
                or not np.all(np.isfinite(z)) or not np.all(np.isfinite(r))
                or not np.allclose(r, r.T) or np.min(np.linalg.eigvalsh(r)) <= 0):
            raise ValueError("Detection requires finite uv and positive-definite 2x2 covariance")
        pred = camera.project(self.x[:3], R_CI)
        if pred is None or not camera.contains(z):
            self.rejected += 1
            return False
        h = camera.jacobian(self.x[:3], R_CI)
        residual = z-pred
        s = h @ self.P @ h.T + r
        self.last_nis = float(residual @ np.linalg.solve(s, residual))
        if self.last_nis > gate:
            self.rejected += 1
            return False
        k = np.linalg.solve(s, h @ self.P).T
        self.x += k @ residual
        ikh = np.eye(6)-k @ h
        self.P = ikh @ self.P @ ikh.T + k @ r @ k.T  # Joseph form
        self.P = (self.P+self.P.T)/2
        self.accepted += 1
        self.last_accepted_s = self.time_s
        return True
