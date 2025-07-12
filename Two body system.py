import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore", message=".*Setting the 'color' property will override the edgecolor or facecolor properties.*")
warnings.filterwarnings("ignore", message=".*At least one element of `rtol` is too small.*")

G = 1.0

def two_body_dynamics(t, y, m1, m2):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = y
    
    dx = x2 - x1
    dy = y2 - y1
    r = np.sqrt(dx**2 + dy**2)
    
    if r < 1e-12:
        r = 1e-12
    
    F = G * m1 * m2 / r**2
    
    ux = dx / r
    uy = dy / r
    
    ax1 = F * ux / m1
    ay1 = F * uy / m1
    ax2 = -F * ux / m2
    ay2 = -F * uy / m2
    
    return [vx1, vy1, vx2, vy2, ax1, ay1, ax2, ay2]

def setup_elliptical_orbit(m1, m2, a, e):
    M = m1 + m2
    mu = G * M
    
    r_peri = a * (1 - e)
    
    r1_bary = m2 * r_peri / M
    r2_bary = m1 * r_peri / M
    
    x1_init = -r1_bary
    y1_init = 0.0
    x2_init = r2_bary
    y2_init = 0.0
    
    v_peri = np.sqrt(mu * (2/r_peri - 1/a))
    
    vx1_init = 0.0
    vy1_init = v_peri * m2 / M
    vx2_init = 0.0
    vy2_init = -v_peri * m1 / M
    
    return [x1_init, y1_init, x2_init, y2_init, 
            vx1_init, vy1_init, vx2_init, vy2_init]

def calculate_conserved_quantities(state, m1, m2):
    x1, y1, x2, y2, vx1, vy1, vx2, vy2 = state
    
    dx = x2 - x1
    dy = y2 - y1
    r = np.sqrt(dx**2 + dy**2)
    
    v1_sq = vx1**2 + vy1**2
    v2_sq = vx2**2 + vy2**2
    
    KE = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
    PE = -G * m1 * m2 / r
    E_total = KE + PE
    
    L = m1 * (x1 * vy1 - y1 * vx1) + m2 * (x2 * vy2 - y2 * vx2)
    
    area_rate = 0.5 * abs(dx * vy2 - dy * vx2)
    
    return E_total, L, area_rate, r, np.sqrt(v1_sq), np.sqrt(v2_sq)

m1_init = 15.0
m2_init = 8.0
a_init = 10.0
e_init = 0.4

fig = plt.figure(figsize=(22, 13))

gs = gridspec.GridSpec(3, 3, width_ratios=[1, 2.5, 1], height_ratios=[1, 1, 1], 
                       wspace=0.7, hspace=0.9)

ax_distance = fig.add_subplot(gs[0, 0])
ax_velocity = fig.add_subplot(gs[1, 0])
ax_energy = fig.add_subplot(gs[2, 0])

ax_orbit = fig.add_subplot(gs[:, 1])

ax_area = fig.add_subplot(gs[0, 2])
ax_kepler2 = fig.add_subplot(gs[1, 2])
ax_conservation = fig.add_subplot(gs[2, 2])

ax_orbit.set_aspect('equal', adjustable='box')
ax_orbit.set_xlim(-15, 15)
ax_orbit.set_ylim(-15, 15)

params = {'m1': m1_init, 'm2': m2_init, 'a': a_init, 'e': e_init}
state = setup_elliptical_orbit(**params)
current_time = 0.0
base_dt = 0.001

initial_energy, initial_L, _, _, _, _ = calculate_conserved_quantities(state, params['m1'], params['m2'])

max_points = 2000
time_data = []
x1_data, y1_data = [], []
x2_data, y2_data = [], []
distance_data = []
velocity1_data, velocity2_data = [], []
energy_data = []
area_rate_data = []
angular_momentum_data = []

energy_errors = []
momentum_errors = []

orbit1_line, = ax_orbit.plot([], [], 'red', linewidth=3, alpha=0.7, label='Body 1 Path')
orbit2_line, = ax_orbit.plot([], [], 'blue', linewidth=3, alpha=0.7, label='Body 2 Path')
cm_line, = ax_orbit.plot([], [], 'green', linewidth=2, alpha=0.6, label='Barycenter')

body1_size = params['a']/12 * (params['m1']/15)**(1/3)
body2_size = params['a']/15 * (params['m2']/15)**(1/3)

body1_circle = plt.Circle((state[0], state[1]), body1_size, facecolor='red', alpha=0.9, zorder=10, edgecolor='darkred', linewidth=2)
body2_circle = plt.Circle((state[2], state[3]), body2_size, facecolor='blue', alpha=0.9, zorder=10, edgecolor='darkblue', linewidth=2)
barycenter_dot = plt.Circle((0, 0), params['a']/50, facecolor='green', alpha=0.8, zorder=10)

ax_orbit.add_patch(body1_circle)
ax_orbit.add_patch(body2_circle)
ax_orbit.add_patch(barycenter_dot)

ax_orbit.set_title('Two-Body Orbital Simulation', 
                   fontsize=16, fontweight='bold', pad=20)
ax_orbit.set_xlabel('x [length units]', fontsize=12)
ax_orbit.set_ylabel('y [length units]', fontsize=12)
ax_orbit.legend(loc='upper right', fontsize=10)
ax_orbit.grid(True, alpha=0.3)

distance_line, = ax_distance.plot([], [], 'purple', linewidth=2)
ax_distance.set_title('Distance vs Time', fontsize=9, pad=20)
ax_distance.set_xlabel('Time', fontsize=8)
ax_distance.set_ylabel('Distance', fontsize=8)
ax_distance.grid(True, alpha=0.3)

vel1_line, = ax_velocity.plot([], [], 'red', linewidth=2, label='Body 1')
vel2_line, = ax_velocity.plot([], [], 'blue', linewidth=2, label='Body 2')
ax_velocity.set_title('Velocities vs Time', fontsize=9, pad=20)
ax_velocity.set_xlabel('Time', fontsize=8)
ax_velocity.set_ylabel('Velocity', fontsize=8)
ax_velocity.legend(fontsize=7)
ax_velocity.grid(True, alpha=0.3)

energy_line, = ax_energy.plot([], [], 'black', linewidth=2)
ax_energy.set_title('Total Energy', fontsize=9, pad=20)
ax_energy.set_xlabel('Time', fontsize=8)
ax_energy.set_ylabel('Energy', fontsize=8)
ax_energy.grid(True, alpha=0.3)

area_line, = ax_area.plot([], [], 'orange', linewidth=2)
ax_area.set_title('Area Sweep Rate', fontsize=9, pad=20)
ax_area.set_xlabel('Time', fontsize=8)
ax_area.set_ylabel('Rate', fontsize=8)
ax_area.grid(True, alpha=0.3)

kepler2_line, = ax_kepler2.plot([], [], 'purple', linewidth=2)
mean_line = ax_kepler2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Mean')
ax_kepler2.set_title('Area Rate Consistency', fontsize=9, pad=20)
ax_kepler2.set_xlabel('Time', fontsize=8)
ax_kepler2.set_ylabel('Area Rate', fontsize=8)
ax_kepler2.legend(fontsize=7)
ax_kepler2.grid(True, alpha=0.3)

momentum_line, = ax_conservation.plot([], [], 'darkgreen', linewidth=2)
ax_conservation.set_title('Angular Momentum', fontsize=9, pad=20)
ax_conservation.set_xlabel('Time', fontsize=8)
ax_conservation.set_ylabel('L', fontsize=8)
ax_conservation.grid(True, alpha=0.3)

plt.subplots_adjust(top=0.85, bottom=0.20, left=0.06, right=0.96)

slider_width = 0.1
slider_height = 0.025
slider_y = 0.95
gap = 0.15

ax_m1 = plt.axes([0.08, slider_y, slider_width, slider_height], facecolor='lightgoldenrodyellow')
ax_m2 = plt.axes([0.08 + slider_width + gap, slider_y, slider_width, slider_height], facecolor='lightgoldenrodyellow')
ax_a = plt.axes([0.08 + 2*(slider_width + gap), slider_y, slider_width, slider_height], facecolor='lightgoldenrodyellow')
ax_e = plt.axes([0.08 + 3*(slider_width + gap), slider_y, slider_width, slider_height], facecolor='lightgoldenrodyellow')

slider_m1 = Slider(ax_m1, 'Mass 1', 1.0, 100.0, valinit=m1_init, valstep=1.0)
slider_m2 = Slider(ax_m2, 'Mass 2', 1.0, 100.0, valinit=m2_init, valstep=1.0)
slider_a = Slider(ax_a, 'Semi-major Axis', 3.0, 25.0, valinit=a_init, valstep=0.5)
slider_e = Slider(ax_e, 'Eccentricity', 0.0, 0.9, valinit=e_init, valstep=0.01)

ax_speed = plt.axes([0.15, 0.08, 0.7, 0.03], facecolor='lightcyan')
speed_slider = Slider(ax_speed, 'Animation Speed', 0.1, 100.0, valinit=1.0, valstep=0.1)
ani = None

def reset_simulation():
    global state, current_time, initial_energy, initial_L, body1_circle, body2_circle, barycenter_dot, ani
    
    params['m1'] = slider_m1.val
    params['m2'] = slider_m2.val
    params['a'] = slider_a.val
    params['e'] = slider_e.val
    
    state = setup_elliptical_orbit(**params)
    current_time = 0.0
    
    initial_energy, initial_L, _, _, _, _ = calculate_conserved_quantities(state, params['m1'], params['m2'])
    
    try:
        patches_to_remove = []
        for patch in ax_orbit.patches:
            patches_to_remove.append(patch)
        
        for patch in patches_to_remove:
            patch.remove()
            
    except Exception as e:
        try:
            while ax_orbit.patches:
                ax_orbit.patches[0].remove()
        except:
            ax_orbit.clear()
            ax_orbit.set_aspect('equal', adjustable='box')
            ax_orbit.set_title('Two-Body Orbital Simulation', 
                              fontsize=16, fontweight='bold', pad=30)
            ax_orbit.set_xlabel('x [length units]', fontsize=12)
            ax_orbit.set_ylabel('y [length units]', fontsize=12)
            ax_orbit.legend(loc='upper right', fontsize=10)
            ax_orbit.grid(True, alpha=0.3)
            
            ax_orbit.add_line(orbit1_line)
            ax_orbit.add_line(orbit2_line)
            ax_orbit.add_line(cm_line)
    
    body1_circle = plt.Circle((state[0], state[1]), params['a']/15 * (params['m1']/15)**(1/3), 
                             facecolor='red', alpha=0.9, zorder=10, 
                             edgecolor='darkred', linewidth=2)
    body2_circle = plt.Circle((state[2], state[3]), params['a']/15 * (params['m2']/15)**(1/3), 
                             facecolor='blue', alpha=0.9, zorder=10,
                             edgecolor='darkblue', linewidth=2)
    barycenter_dot = plt.Circle((0, 0), params['a']/50, 
                               facecolor='green', alpha=0.8, zorder=10)
    
    ax_orbit.add_patch(body1_circle)
    ax_orbit.add_patch(body2_circle)
    ax_orbit.add_patch(barycenter_dot)
    
    max_extent = params['a'] * (1 + params['e']) * 1.2
    ax_orbit.set_xlim(-max_extent, max_extent)
    ax_orbit.set_ylim(-max_extent, max_extent)
    
    time_data.clear()
    x1_data.clear()
    y1_data.clear()
    x2_data.clear()
    y2_data.clear()
    distance_data.clear()
    velocity1_data.clear()
    velocity2_data.clear()
    energy_data.clear()
    area_rate_data.clear()
    angular_momentum_data.clear()
    energy_errors.clear()
    momentum_errors.clear()

def update_params(val):
    reset_simulation()

slider_m1.on_changed(update_params)
slider_m2.on_changed(update_params)
slider_a.on_changed(update_params)
slider_e.on_changed(update_params)

def animate(frame):
    global state, current_time
    
    dt = base_dt * speed_slider.val
    
    sol = solve_ivp(two_body_dynamics, (current_time, current_time + dt), 
                    state, args=(params['m1'], params['m2']),
                    rtol=1e-12, atol=1e-14, method='DOP853')
    
    if sol.success:
        state = sol.y[:, -1]
        current_time += dt
        
        E_total, L, area_rate, r, v1, v2 = calculate_conserved_quantities(state, params['m1'], params['m2'])
        
        x1, y1, x2, y2 = state[:4]
        
        time_data.append(current_time)
        x1_data.append(x1)
        y1_data.append(y1)
        x2_data.append(x2)
        y2_data.append(y2)
        distance_data.append(r)
        velocity1_data.append(v1)
        velocity2_data.append(v2)
        energy_data.append(E_total)
        area_rate_data.append(area_rate)
        angular_momentum_data.append(L)
        
        energy_error = abs(E_total - initial_energy) / abs(initial_energy) if initial_energy != 0 else 0
        momentum_error = abs(L - initial_L) / abs(initial_L) if initial_L != 0 else 0
        energy_errors.append(energy_error)
        momentum_errors.append(momentum_error)
        
        if len(time_data) > max_points:
            time_data.pop(0)
            x1_data.pop(0)
            y1_data.pop(0)
            x2_data.pop(0)
            y2_data.pop(0)
            distance_data.pop(0)
            velocity1_data.pop(0)
            velocity2_data.pop(0)
            energy_data.pop(0)
            area_rate_data.pop(0)
            angular_momentum_data.pop(0)
            energy_errors.pop(0)
            momentum_errors.pop(0)
        
        orbit1_line.set_data(x1_data, y1_data)
        orbit2_line.set_data(x2_data, y2_data)
        
        M = params['m1'] + params['m2']
        cm_x = [(params['m1']*xx + params['m2']*xxx)/M 
                for xx, xxx in zip(x1_data, x2_data)]
        cm_y = [(params['m1']*yy + params['m2']*yyy)/M 
                for yy, yyy in zip(y1_data, y2_data)]
        cm_line.set_data(cm_x, cm_y)
        
        try:
            body1_circle.center = (x1, y1)
            body2_circle.center = (x2, y2)
            barycenter_dot.center = (cm_x[-1] if cm_x else 0, cm_y[-1] if cm_y else 0)
        except:
            pass
        
        distance_line.set_data(time_data, distance_data)
        vel1_line.set_data(time_data, velocity1_data)
        vel2_line.set_data(time_data, velocity2_data)
        energy_line.set_data(time_data, energy_data)
        area_line.set_data(time_data, area_rate_data)
        kepler2_line.set_data(time_data, area_rate_data)
        momentum_line.set_data(time_data, angular_momentum_data)
        
        if area_rate_data:
            mean_area = np.mean(area_rate_data)
            mean_line.set_ydata([mean_area, mean_area])
        
        for ax in [ax_distance, ax_velocity, ax_energy, 
                   ax_area, ax_kepler2, ax_conservation]:
            ax.relim()
            ax.autoscale_view()
    
    return_objects = [orbit1_line, orbit2_line, cm_line, distance_line, 
                     vel1_line, vel2_line, energy_line, area_line, 
                     kepler2_line, momentum_line]
    
    try:
        if body1_circle and hasattr(body1_circle, 'center'):
            return_objects.append(body1_circle)
        if body2_circle and hasattr(body2_circle, 'center'):
            return_objects.append(body2_circle)
        if barycenter_dot and hasattr(barycenter_dot, 'center'):
            return_objects.append(barycenter_dot)
    except:
        pass
    
    return tuple(return_objects)

ani = FuncAnimation(fig, animate, frames=100000, interval=25, blit=False, repeat=True)

plt.show()
