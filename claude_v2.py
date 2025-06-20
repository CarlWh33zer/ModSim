import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------
# GEOMETRIE & SICHTLINIEN
# ---------------------------------------------------------------------
A, B = 8000.0, 9000.0                    # Ellipsen-Halbachsen  [mm]
ALPHA = np.deg2rad(65.0)                 # Keil-Öffnung         [rad]
THETA_WEDGE = np.deg2rad(180.0)          # Keil-Mittelachse     [rad]

def in_wedge(theta: float) -> bool:
    """True, falls Polarwinkel θ im Keil liegt."""
    delta = (theta - THETA_WEDGE + np.pi) % (2*np.pi) - np.pi
    return np.abs(delta) <= ALPHA/2

def inside_room(pt: np.ndarray) -> bool:
    """Punktprüfung: Ellipse UND nicht im Keil."""
    x, y = pt
    theta = np.arctan2(y, x)
    return (x**2)/B**2 + (y**2)/A**2 <= 1.0 and not in_wedge(theta)

def on_ellipse(theta: float) -> np.ndarray:
    """Perimeterpunkt für Winkel θ (falls außerhalb Keils)."""
    assert not in_wedge(theta), "θ im Keil!"
    return np.array([B*np.cos(theta), A*np.sin(theta)])

def visible(sensor: np.ndarray, target: np.ndarray, n_samples: int = 100) -> bool:
    """Abtastung der Geraden  z → y  (LoS-Test)."""
    ts = np.linspace(0, 1, n_samples)
    segment = sensor + ts[:, None] * (target - sensor)
    return all(inside_room(p) for p in segment)

# ---------------------------------------------------------------------
# SENSORRAUSCHEN  σ(r)  [mm]
# ---------------------------------------------------------------------
def sensor_noise_std(sensor_id: int, distance_mm: float) -> float:
    """Gerätespezifisches σ(r) gem. Aufgabenstellung."""
    if sensor_id == 0:      # Gerät 1
        return 2.5 + 0.0010 * distance_mm
    if sensor_id == 1:      # Gerät 2
        return 5.0 + 0.0005 * distance_mm
    if sensor_id == 2:      # Gerät 3
        return 0.5 + 0.0020 * distance_mm
    raise ValueError("Ungültige sensor_id")

# Umrechnung auf Gewicht  w_i = 1/σ_i²
def measurement_weights(sensor_ids, distances_mm):
    sigmas = [sensor_noise_std(i, r) for i, r in zip(sensor_ids, distances_mm)]
    return np.diag(1.0 / (np.array(sigmas) ** 2))

# ---------------------------------------------------------------------
# RESIDUEN  f(y)  und JACOBIAN  A
# ---------------------------------------------------------------------
def compute_residuals(y, sensors, r_meas):
    """f_i = ||y - z_i|| - r_i"""
    return np.linalg.norm(y - sensors, axis=1) - r_meas

def compute_jacobian(y, sensors, eps: float = 1e-12):
    """A_{i,:} = (y - z_i)/||y - z_i||"""
    diffs = y - sensors
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms[norms < eps] = np.inf           # Singularitäts­schutz
    return diffs / norms

# ---------------------------------------------------------------------
# GEWICHTETER GAUSS-NEWTON
# ---------------------------------------------------------------------
def weighted_gauss_newton(y0, sensors, r_meas, sensor_ids,
                          max_iter: int = 20, tol: float = 1e-8):
    """Nichtlineare WLS-Schätzung der Position."""
    y = y0.copy()
    W = measurement_weights(sensor_ids, r_meas)     # konst. Gewichte
    for _ in range(max_iter):
        f  = compute_residuals(y, sensors, r_meas)
        A  = compute_jacobian(y, sensors)
        lhs = A.T @ W @ A
        rhs = -A.T @ W @ f
        try:
            dy = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:   # schlecht konditioniert
            break
        y += dy
        if np.linalg.norm(dy) < tol:
            break
    return y

# ---------------------------------------------------------------------
# RMS-BERECHNUNG AN EINEM ZIELPUNKT
# ---------------------------------------------------------------------
def rms_error(target, sensors, n_mc: int = 200):
    """Monte-Carlo-basierter RMS-Fehler an Position target."""
    vis_idx = [i for i, s in enumerate(sensors) if visible(s, target)]
    if len(vis_idx) < 2:                         # < 2 Sichtlinien ⇒ Strafwert
        return np.hypot(A, B)

    S   = sensors[vis_idx]
    r0  = np.linalg.norm(target - S, axis=1)
    errs = []
    for _ in range(n_mc):
        r_noisy = np.array([
            r + np.random.normal(0, sensor_noise_std(i, r))
            for i, r in zip(vis_idx, r0)
        ])
        y_hat = weighted_gauss_newton(target, S, r_noisy, vis_idx)
        errs.append(np.linalg.norm(y_hat - target) ** 2)
    return np.sqrt(np.mean(errs))

# ---------------------------------------------------------------------
# GLOBALER QUALITÄTSSCORE  q₀.₉₅
# ---------------------------------------------------------------------
def quality_q95(layout, grid_pts, n_mc=100, n_jobs=8):
    errs = Parallel(n_jobs=n_jobs)(
        delayed(rms_error)(p, layout, n_mc) for p in grid_pts
    )
    return np.percentile(errs, 95.0), np.array(errs)


# ---------------------------------------------------------------------
# SENSOR-LAYOUT-OPTIMIERUNG
# ---------------------------------------------------------------------
def optimize_layout(grid, n_trials=300, n_refine=50, n_jobs=8):
    # Kandidatenpunkte (1/720 ≈ 0.5°-Raster) ausserhalb Keils
    angles = [th for th in np.linspace(-np.pi, np.pi, 720, endpoint=False)
              if not in_wedge(th)]
    perimeter = np.vstack([on_ellipse(th) for th in angles])

    # ---- PHASE 1: Random Search -------------------------------------
    def random_layout():
        idx = np.random.choice(len(perimeter), 3, replace=False)
        Z   = perimeter[idx]
        q95, _ = quality_q95(Z, grid, n_mc=50, n_jobs=2)
        return q95, Z

    best_q95, best_Z = min(
        Parallel(n_jobs=n_jobs)(delayed(random_layout)() for _ in trange(n_trials)),
        key=lambda t: t[0]
    )

    # ---- PHASE 2: Lokale Verfeinerung -------------------------------
    for _ in range(n_refine):
        s_idx = np.random.randint(3)
        cur_th = np.arctan2(best_Z[s_idx,1], best_Z[s_idx,0])
        cand_th = cur_th + np.random.normal(0, 0.1)    # ~5.7°
        if in_wedge(cand_th):
            continue
        test_Z = best_Z.copy()
        test_Z[s_idx] = on_ellipse(cand_th)
        test_q95, _ = quality_q95(test_Z, grid, n_mc=75, n_jobs=n_jobs)
        if test_q95 < best_q95:
            best_q95, best_Z = test_q95, test_Z
            print(f"  Verbesserung: q95 = {best_q95:.2f} mm")
    return best_Z, best_q95


# ---------------------------------------------------------------------
# VISUALISIERUNG
# ---------------------------------------------------------------------
def plot_layout(ax, sensors, rms, grid):
    # Interpolierte Heatmap
    xi = np.linspace(-B, B, 150)
    yi = np.linspace(-A, A, 150)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((grid[:,0], grid[:,1]), rms, (xi, yi), method='linear')
    mask = (xi**2)/B**2 + (yi**2)/A**2 > 1.0
    zi = np.ma.masked_where(mask | np.isnan(zi) | (zi > 1e3), zi)
    im = ax.contourf(xi, yi, zi, 20, cmap='viridis')

    # Ellipse + Keil
    th = np.linspace(-np.pi, np.pi, 400)
    ax.plot(B*np.cos(th), A*np.sin(th), 'k', lw=2)
    for sign in (-1,+1):
        edge = THETA_WEDGE + sign*ALPHA/2
        ax.plot([0,B*np.cos(edge)], [0,A*np.sin(edge)], 'k')

    # Sensoren
    for i,p in enumerate(sensors):
        ax.scatter(*p, marker='^', s=120, edgecolors='k',
                   color=['r','b','g'][i], zorder=5)
        ax.annotate(f'S{i+1}', p, textcoords='offset points', xytext=(5,5),
                    weight='bold')
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    return im