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

# =============================================================================
# GEOMETRIE-PARAMETER (Aufgabe 3.2) - Gruppe 4
# =============================================================================

A, B = 8000.0, 9000.0  # Halbachsen [mm] - vertikal, horizontal
ALPHA = np.deg2rad(65.0)  # Keilwinkel
THETA_WEDGE = np.deg2rad(180.0)  # Keil-Mittelpunkt (180° = links)

def in_wedge(theta):
    """Prüft ob Winkel theta im verbotenen Keil liegt"""
    delta = (theta - THETA_WEDGE + np.pi) % (2*np.pi) - np.pi
    return np.abs(delta) <= ALPHA/2

def inside_room(point):
    """Prüft ob Punkt im zugänglichen Raumbereich liegt"""
    x, y = point
    theta = np.arctan2(y, x)
    # Ellipsen-Bedingung UND nicht im Keil
    return (x**2)/B**2 + (y**2)/A**2 <= 1.0 and not in_wedge(theta)

def on_ellipse(theta):
    """Berechnet Punkt auf Ellipsenrand für gegebenen Winkel"""
    assert not in_wedge(theta), f"Winkel {np.degrees(theta):.1f}° liegt im verbotenen Keil!"
    return np.array([B*np.cos(theta), A*np.sin(theta)])

# =============================================================================
# SENSOR-SPEZIFISCHE RAUSCHKENNLINIEN (Aufgabe 3.2)
# =============================================================================

def sensor_noise_std(sensor_id, distance):
    """
    Rauschkennlinien σ(r) für die 3 Messgeräte [mm]
    """
    if sensor_id == 0:      # Messgerät 1
        return 2.5 + 0.0010 * distance
    elif sensor_id == 1:    # Messgerät 2
        return 5.0 + 0.0005 * distance
    elif sensor_id == 2:    # Messgerät 3
        return 0.5 + 0.0020 * distance
    else:
        raise ValueError(f"Ungültige sensor_id: {sensor_id}")

# =============================================================================
# SICHTBARKEITS-PRÜFUNG (Wandabschattung)
# =============================================================================

def visible(sensor_pos, target_pos, n_samples=100):
    """
    Prüft ob Sensor das Ziel sehen kann (keine Wandabschattung)
    
    KORREKTUR: Robustere Sichtlinienprüfung
    """
    # Sichtlinie zwischen Sensor und Ziel abtasten
    t_values = np.linspace(0, 1, n_samples)
    sight_line = sensor_pos + t_values[:,None] * (target_pos - sensor_pos)
    
    # Alle Punkte der Sichtlinie müssen im zugänglichen Raum liegen
    return all(inside_room(point) for point in sight_line)

# =============================================================================
# MATHEMATISCHE GRUNDLAGEN: RESIDUEN UND JACOBIAN (Aufgabe 3.1)
# =============================================================================

def compute_residuals(position_estimate, sensor_positions, measured_distances):
    """
    Berechnet Residuen-Vektor f(y) = ||y - zᵢ|| - rᵢ
    """
    predicted_distances = np.linalg.norm(position_estimate - sensor_positions, axis=1)
    residuals = predicted_distances - measured_distances
    return residuals

def compute_jacobian(position_estimate, sensor_positions, tolerance=1e-12):
    """
    Berechnet Jacobian-Matrix A mit Aᵢ = ∂fᵢ/∂y = (y - zᵢ)/||y - zᵢ||
    """
    differences = position_estimate - sensor_positions
    norms = np.linalg.norm(differences, axis=1)
    
    jacobian_matrix = np.zeros_like(differences)
    valid_indices = norms > tolerance
    jacobian_matrix[valid_indices, :] = differences[valid_indices, :] / norms[valid_indices, None]
    
    return jacobian_matrix

# =============================================================================
# GEWICHTETER GAUSS-NEWTON-ALGORITHMUS (Aufgabe 3.1)
# =============================================================================

def weighted_gauss_newton(initial_estimate, sensor_positions, measured_distances, 
                         sensor_ids, max_iterations=20, convergence_tolerance=1e-8):
    """
    Gewichteter Gauss-Newton-Algorithmus für Multilateration
    
    KORREKTUR: Gewichte basieren auf gemessenen Distanzen, nicht auf aktueller Schätzung!
    """
    current_estimate = initial_estimate.copy()
    
    # WICHTIG: Gewichte basieren auf gemessenen Distanzen rᵢ
    weights = np.array([1.0 / sensor_noise_std(sid, r_measured)**2 
                       for sid, r_measured in zip(sensor_ids, measured_distances)])
    weight_matrix = np.diag(weights)
    
    for iteration in range(max_iterations):
        residual_vector = compute_residuals(current_estimate, sensor_positions, measured_distances)
        jacobian_matrix = compute_jacobian(current_estimate, sensor_positions)
        
        # Gewichtete Normalgleichungen: (AᵀWA) Δy = -AᵀWf
        lhs_matrix = jacobian_matrix.T @ weight_matrix @ jacobian_matrix
        rhs_vector = -jacobian_matrix.T @ weight_matrix @ residual_vector
        
        try:
            position_update = np.linalg.solve(lhs_matrix, rhs_vector)
        except np.linalg.LinAlgError:
            break
            
        current_estimate += position_update
        
        if np.linalg.norm(position_update) < convergence_tolerance:
            break
    
    return current_estimate

# =============================================================================
# MONTE-CARLO-SIMULATION FÜR RMS-FEHLER
# =============================================================================

def compute_rms_error_at_target(target_position, sensor_positions, n_monte_carlo=200):
    """
    Berechnet RMS-Positionsfehler an einem Zielpunkt via Monte-Carlo-Simulation
    
    KORREKTUR: Berücksichtigt sensor-spezifische Rauschkennlinien korrekt
    """
    # Sichtbare Sensoren identifizieren
    visible_sensor_indices = [i for i, sensor_pos in enumerate(sensor_positions) 
                             if visible(sensor_pos, target_position)]
    
    # Mindestens 2 Sensoren erforderlich für 2D-Positionierung
    if len(visible_sensor_indices) < 2:
        return np.hypot(A, B)  # Penalty: Maximale Distanz in Ellipse
    
    # Nur sichtbare Sensoren verwenden
    visible_sensors = sensor_positions[visible_sensor_indices]
    true_distances = np.linalg.norm(target_position - visible_sensors, axis=1)
    
    # Monte-Carlo-Simulation
    squared_errors = []
    
    for mc_iteration in range(n_monte_carlo):
        # Sensor-spezifisches Messrauschen generieren
        noisy_distances = np.zeros(len(true_distances))
        
        for i, (sensor_idx, true_dist) in enumerate(zip(visible_sensor_indices, true_distances)):
            noise_std = sensor_noise_std(sensor_idx, true_dist)
            noise_sample = np.random.normal(0, noise_std)
            noisy_distances[i] = true_dist + noise_sample
        
        # Gewichtete Multilateration
        estimated_position = weighted_gauss_newton(
            initial_estimate=target_position,  # Gute Startschätzung für MC
            sensor_positions=visible_sensors,
            measured_distances=noisy_distances,
            sensor_ids=visible_sensor_indices
        )
        
        # Quadratischer Positionsfehler
        position_error = np.linalg.norm(estimated_position - target_position)
        squared_errors.append(position_error**2)
    
    # RMS-Fehler: sqrt(E[||ŷ - y||²])
    rms_error = np.sqrt(np.mean(squared_errors))
    
    return rms_error

# =============================================================================
# GRID-GENERIERUNG (KORRIGIERT: nur zugängliche Punkte)
# =============================================================================

def generate_measurement_grid(n_x=60, n_y=60):
    """
    Erzeugt hochauflösendes Messgitter im zugänglichen Raumbereich
    
    KORREKTUR: Nur Punkte im zugänglichen Raum (nicht im Keil)
    """
    x_coords = np.linspace(-B, B, n_x, endpoint=False) + 0.5*(2*B/n_x)
    y_coords = np.linspace(-A, A, n_y, endpoint=False) + 0.5*(2*A/n_y)
    
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    all_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    
    # NUR zugängliche Punkte (Ellipse ohne Keil)
    valid_mask = np.array([inside_room(point) for point in all_points])
    valid_points = all_points[valid_mask]
    
    print(f"Messgitter generiert: {len(valid_points)} gültige Punkte")
    return valid_points

# =============================================================================
# LAYOUT-BEWERTUNG MIT PARALLELISIERUNG
# =============================================================================

def evaluate_sensor_layout(sensor_positions, measurement_grid, n_monte_carlo=100, n_jobs=8):
    """
    Bewertet Sensor-Layout über gesamten Messbereich
    """
    rms_values = Parallel(n_jobs=n_jobs)(
        delayed(compute_rms_error_at_target)(point, sensor_positions, n_monte_carlo)
        for point in measurement_grid
    )
    
    rms_array = np.array(rms_values)
    q95_metric = np.percentile(rms_array, 95.0)
    
    return q95_metric, rms_array

# =============================================================================
# OPTIMIERUNGSALGORITHMUS (VERBESSERT)
# =============================================================================

def optimize_sensor_layout(n_trials=300, measurement_grid=None, n_local_refinements=50, n_jobs=8):
    """
    Verbesserte Sensor-Layout-Optimierung
    
    KORREKTUR: Systematischere Optimierung mit lokaler Verfeinerung
    """
    if measurement_grid is None:
        measurement_grid = generate_measurement_grid(n_x=40, n_y=40)
    
    # Gültige Kandidatenpositionen auf Ellipsenrand (ohne Keil)
    candidate_angles = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    valid_angles = [angle for angle in candidate_angles if not in_wedge(angle)]
    candidate_positions = np.array([on_ellipse(angle) for angle in valid_angles])
    
    print(f"Verfügbare Kandidatenpositionen: {len(candidate_positions)}")
    print(f"Starte Random Search mit {n_trials} Versuchen...")

    # Phase 1: Random Search
    def evaluate_random_layout():
        selected_indices = np.random.choice(len(candidate_positions), 3, replace=False)
        layout = candidate_positions[selected_indices]
        q95, _ = evaluate_sensor_layout(layout, measurement_grid, n_monte_carlo=50, n_jobs=2)
        return q95, layout

    random_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_random_layout)() for _ in trange(n_trials)
    )

    optimal_q95, optimal_positions = min(random_results, key=lambda x: x[0])
    print(f"Bestes Ergebnis nach Random Search: Q95 = {optimal_q95:.2f} mm")

    # Phase 2: Lokale Verfeinerung
    print("Starte lokale Verfeinerung...")
    for refinement_step in range(n_local_refinements):
        # Zufälligen Sensor auswählen
        sensor_idx = np.random.randint(3)
        current_angle = np.arctan2(optimal_positions[sensor_idx][1], optimal_positions[sensor_idx][0])
        
        # Kleine Winkelverschiebung
        angle_perturbation = np.random.normal(0, 0.1)  # ~5.7°
        new_angle = current_angle + angle_perturbation
        
        if not in_wedge(new_angle):
            # Teste neue Position
            test_positions = optimal_positions.copy()
            test_positions[sensor_idx] = on_ellipse(new_angle)
            
            test_q95, _ = evaluate_sensor_layout(test_positions, measurement_grid, 
                                               n_monte_carlo=75, n_jobs=n_jobs)
            
            if test_q95 < optimal_q95:
                optimal_q95 = test_q95
                optimal_positions = test_positions.copy()
                print(f"  Lokale Verbesserung: Q95 = {test_q95:.2f} mm")

    return optimal_positions, optimal_q95

# =============================================================================
# VISUALISIERUNG
# =============================================================================

def add_elliptical_wedge(ax, a, b, theta_center, theta_width, n=100):
    """Fügt elliptischen Keil als Patch hinzu"""
    theta1 = theta_center - theta_width / 2
    theta2 = theta_center + theta_width / 2
    thetas = np.linspace(theta1, theta2, n)

    arc = np.column_stack((b * np.cos(thetas), a * np.sin(thetas)))
    vertices = np.vstack([[0, 0], arc, [0, 0]])
    codes = [Path.MOVETO] + [Path.LINETO]*(len(arc)) + [Path.CLOSEPOLY]

    path = Path(vertices, codes)
    patch = PathPatch(path, facecolor='white', edgecolor='black', lw=1.5, zorder=5)
    ax.add_patch(patch)

def plot_layout(ax, sensors, rms_vals, grid_pts, a, b, wedge_center, wedge_angle,
                cmap='viridis', mode='contour'):
    """Zeichnet Layout mit RMS-Visualisierung"""
    
    cmap_used = plt.get_cmap(cmap).copy()
    cmap_used.set_bad(color='lightgray')

    if mode == 'contour':
        # Interpolierte Konturkarte
        x, y = grid_pts[:, 0], grid_pts[:, 1]
        rms = rms_vals

        xi = np.linspace(np.min(x), np.max(x), 150)
        yi = np.linspace(np.min(y), np.max(y), 150)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((x, y), rms, (xi, yi), method='linear')
        ellipse_mask = (xi**2 / b**2 + yi**2 / a**2) <= 1.0
        zi = np.ma.masked_where(~ellipse_mask | np.isnan(zi) | (zi > 1000), zi)

        sc = ax.contourf(xi, yi, zi, levels=20, cmap=cmap_used, zorder=2)
    else:
        # Scatter-Plot
        rms_masked = np.ma.masked_where(rms_vals > 1000, rms_vals)
        sc = ax.scatter(grid_pts[:, 0], grid_pts[:, 1],
                        c=rms_masked, cmap=cmap_used,
                        s=25, marker='s', edgecolors='none', zorder=2)

    # Keil und Ellipsenrand
    add_elliptical_wedge(ax, a, b, wedge_center, wedge_angle)
    
    th = np.linspace(-np.pi, np.pi, 400)
    ax.plot(b * np.cos(th), a * np.sin(th), 'k-', lw=2, zorder=6)

    # Sensoren
    colors = ['red', 'blue', 'green']
    for idx, p in enumerate(sensors):
        ax.scatter(p[0], p[1], marker='^', s=150,
                   edgecolor='black', facecolor=colors[idx],
                   linewidth=2, zorder=7)
        ax.annotate(f'S{idx+1}', (p[0], p[1]), xytext=(5, 5),
                    textcoords='offset points', fontweight='bold', zorder=8)

    ax.set_aspect('equal')
    ax.set_xlim(-b * 1.1, b * 1.1)
    ax.set_ylim(-a * 1.1, a * 1.1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')

    return sc


# =============================================================================
# HAUPTAUSFÜHRUNG
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)  # Reproduzierbarkeit
    
    print("=== MULTILATERATION-OPTIMIERUNG ===")
    print(f"Ellipse: A={A}mm (vertikal), B={B}mm (horizontal)")
    print(f"Keilwinkel: {np.degrees(ALPHA):.1f}°")
    print(f"Keil-Position: {np.degrees(THETA_WEDGE):.1f}° (links)")
    print()

    # Messgitter generieren (>500 Punkte wie gefordert)
    GRID = generate_measurement_grid(n_x=20, n_y=20)
    print(f"Anzahl Messpunkte: {len(GRID)} (Anforderung: >500)")
    
    # Optimierung durchführen
    print("\nStarte Optimierung...")
    best_sensors, best_q95 = optimize_sensor_layout(n_trials=50, measurement_grid=GRID, n_local_refinements=25)
    
    print(f"\n=== OPTIMIERUNGSERGEBNISSE ===")
    print("Optimales Sensor-Layout [mm]:")
    for i, pos in enumerate(best_sensors):
        angle = np.degrees(np.arctan2(pos[1], pos[0]))
        dist_to_center = np.linalg.norm(pos)
        print(f"  Sensor {i+1}: ({pos[0]:8.1f}, {pos[1]:8.1f}) - Winkel: {angle:6.1f}° - Abstand: {dist_to_center:.1f}mm")
    print(f"95%-Quantil RMS-Fehler: {best_q95:.2f} mm")

    # Ungünstige Vergleichslayouts erstellen
    print("\nErzeuge Vergleichslayouts für Bewertung...")
    
    # Layout 1: Sensoren clustered (schlecht für Triangulation)
    bad_layout_1 = np.array([
        on_ellipse(np.deg2rad(30)),   # Rechts oben
        on_ellipse(np.deg2rad(40)),   # Rechts oben (nahe)
        on_ellipse(np.deg2rad(50))    # Rechts oben (nahe)
    ])
    
    # Layout 2: Sensoren auf einer Seite (noch schlechter)
    bad_layout_2 = np.array([
        on_ellipse(np.deg2rad(-30)),  # Rechts unten
        on_ellipse(np.deg2rad(-40)),  # Rechts unten (nahe)
        on_ellipse(np.deg2rad(-50))   # Rechts unten (nahe)
    ])
    
    # RMS-Werte für alle Layouts berechnen
    print("Berechne finale RMS-Bewertung für alle Layouts...")
    _, rms_best = evaluate_sensor_layout(best_sensors, GRID, n_monte_carlo=100, n_jobs=8)
    _, rms_bad1 = evaluate_sensor_layout(bad_layout_1, GRID, n_monte_carlo=100, n_jobs=8)
    _, rms_bad2 = evaluate_sensor_layout(bad_layout_2, GRID, n_monte_carlo=100, n_jobs=8)
    
    # Qualitätsbewertung
    q95_bad1 = np.percentile(rms_bad1, 95.0)
    q95_bad2 = np.percentile(rms_bad2, 95.0)
    
    print(f"\n=== LAYOUT-VERGLEICH ===")
    print(f"Optimales Layout:     Q95 = {best_q95:.1f} mm")
    print(f"Schlechtes Layout 1:  Q95 = {q95_bad1:.1f} mm  (Faktor {q95_bad1/best_q95:.1f}×)")
    print(f"Schlechtes Layout 2:  Q95 = {q95_bad2:.1f} mm  (Faktor {q95_bad2/best_q95:.1f}×)")
    
    # Zusätzliche Statistiken
    print(f"\n=== DETAILIERTE STATISTIKEN ===")
    print(f"Optimales Layout:")
    print(f"  Mittelwert RMS: {np.mean(rms_best):.1f} mm")
    print(f"  Median RMS:     {np.median(rms_best):.1f} mm")
    print(f"  Max RMS:        {np.max(rms_best):.1f} mm")
    print(f"  90%-Quantil:    {np.percentile(rms_best, 90):.1f} mm")


    # Visualisierung
def create_visualization():
    print("\nErstelle Visualisierung...")
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    cax = fig.add_subplot(gs[3])
    sc1 = plot_layout(ax1, best_sensors, rms_best, GRID, A, B, THETA_WEDGE, ALPHA, mode='contour')
    ax1.set_title(f'Optimales Layout\nQ95 = {best_q95:.1f} mm', fontsize=12, fontweight='bold')
    sc2 = plot_layout(ax2, bad_layout_1, rms_bad1, GRID, A, B, THETA_WEDGE, ALPHA, mode='contour')
    ax2.set_title(f'Ungünstiges Layout 1\nQ95 = {q95_bad1:.1f} mm', fontsize=12)
    sc3 = plot_layout(ax3, bad_layout_2, rms_bad2, GRID, A, B, THETA_WEDGE, ALPHA, mode='contour')
    ax3.set_title(f'Ungünstiges Layout 2\nQ95 = {q95_bad2:.1f} mm', fontsize=12)
    # Gemeinsame Farbleiste
    cbar = fig.colorbar(sc1, cax=cax)
    cbar.set_label('RMS-Positionsfehler [mm]', fontsize=11)
    plt.suptitle('Multilateration: Sensor-Layout Optimierung (Gruppe 4)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
create_visualization()