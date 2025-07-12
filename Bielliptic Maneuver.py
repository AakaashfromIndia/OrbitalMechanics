import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore", message=".*Setting the 'color' property will override the edgecolor or facecolor properties.*")
warnings.filterwarnings("ignore", message=".*At least one element of `rtol` is too small.*")

G = 1.0

def spacecraft_dynamics(t, y, M_central):
    x, y_pos, vx, vy = y
    
    r = np.sqrt(x**2 + y_pos**2)
    
    if r < 1e-12:
        r = 1e-12
    
    ax = -G * M_central * x / r**3
    ay = -G * M_central * y_pos / r**3
    
    return [vx, vy, ax, ay]

def setup_circular_orbit(r, M_central, angle=0):
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    
    v_circular = np.sqrt(G * M_central / r)
    
    vx = -v_circular * np.sin(angle)
    vy = v_circular * np.cos(angle)
    
    return [x, y, vx, vy]

def setup_bielliptic_transfer(r1, r2, r3, M_central, phase=0):
    """
    Setup bielliptic transfer from r1 to r2 via intermediate radius r3
    r1: initial circular orbit radius
    r2: final circular orbit radius  
    r3: intermediate apoapsis radius (should be > max(r1, r2))
    """
    # Ensure r3 is larger than both r1 and r2 for a proper bielliptic transfer
    if r3 <= max(r1, r2):
        r3 = max(r1, r2) * 1.5
    
    # Circular velocities
    v1_circular = np.sqrt(G * M_central / r1)
    v2_circular = np.sqrt(G * M_central / r2)
    
    # First transfer ellipse: from r1 to r3
    a1_transfer = (r1 + r3) / 2
    v1_transfer = np.sqrt(G * M_central * (2/r1 - 1/a1_transfer))
    v3_transfer1 = np.sqrt(G * M_central * (2/r3 - 1/a1_transfer))
    
    # Second transfer ellipse: from r3 to r2
    a2_transfer = (r3 + r2) / 2
    v3_transfer2 = np.sqrt(G * M_central * (2/r3 - 1/a2_transfer))
    v2_transfer = np.sqrt(G * M_central * (2/r2 - 1/a2_transfer))
    
    # Delta-V calculations
    delta_v1 = v1_transfer - v1_circular  # First burn at r1
    delta_v2 = v3_transfer2 - v3_transfer1  # Second burn at r3
    delta_v3 = v2_circular - v2_transfer  # Third burn at r2
    
    # Transfer times
    transfer_time1 = np.pi * np.sqrt(a1_transfer**3 / (G * M_central))
    transfer_time2 = np.pi * np.sqrt(a2_transfer**3 / (G * M_central))
    total_transfer_time = transfer_time1 + transfer_time2
    
    # Initial state
    angle = phase
    x = r1 * np.cos(angle)
    y = r1 * np.sin(angle)
    
    vx = -v1_transfer * np.sin(angle)
    vy = v1_transfer * np.cos(angle)
    
    return ([x, y, vx, vy], delta_v1, delta_v2, delta_v3, 
            transfer_time1, transfer_time2, total_transfer_time,
            a1_transfer, a2_transfer, v1_circular, v2_circular, r3)

def calculate_orbital_elements(state, M_central):
    x, y, vx, vy = state
    
    r = np.sqrt(x**2 + y**2)
    v = np.sqrt(vx**2 + vy**2)
    
    energy = 0.5 * v**2 - G * M_central / r
    
    h = x * vy - y * vx
    
    if energy < 0:
        a = -G * M_central / (2 * energy)
    else:
        a = float('inf')
    
    if energy < 0:
        e = np.sqrt(1 + 2 * energy * h**2 / (G * M_central)**2)
    else:
        e = float('inf')
    
    return energy, h, a, e, r, v

def is_near_event(current_time, event_time, threshold=0.2):
    return abs(current_time - event_time) < threshold

M_central_init = 100.0
r1_init = 5.0
r2_init = 10.0
r3_init = 15.0  # Intermediate radius for bielliptic transfer
phase_init = 0.0

fig = plt.figure(figsize=(24, 16))
gs = gridspec.GridSpec(4, 4, width_ratios=[1, 2, 1, 1], height_ratios=[1.1, 1.1, 1.1, 0.3], 
                       wspace=0.3, hspace=1)

ax_orbit = fig.add_subplot(gs[0:3, 1])
ax_orbit.set_aspect('equal', adjustable='box')

ax_altitude = fig.add_subplot(gs[0, 0])
ax_velocity = fig.add_subplot(gs[1, 0])
ax_energy = fig.add_subplot(gs[2, 0])

ax_phase = fig.add_subplot(gs[0, 2])
ax_events = fig.add_subplot(gs[1, 2])
ax_delta_v = fig.add_subplot(gs[2, 2])

ax_mission = fig.add_subplot(gs[:, 3])

params = {
    'M_central': M_central_init,
    'r1': r1_init,
    'r2': r2_init,
    'r3': r3_init,
    'phase': phase_init
}

current_time = 0.0
base_dt = 0.01
max_points = 3000
simulation_mode = 'transfer1'
transfer_start_time = 0.0

event_times = {}
event_markers = []
highlighted_events = []

time_data = []
x_data, y_data = [], []
altitude_data = []
velocity_data = []
energy_data = []

# Initialize bielliptic transfer
(transfer_state, delta_v1, delta_v2, delta_v3, 
 transfer_time1, transfer_time2, total_transfer_time,
 a1_transfer, a2_transfer, v1_circular, v2_circular, r3_actual) = setup_bielliptic_transfer(
    params['r1'], params['r2'], params['r3'], params['M_central'], params['phase'])

state = transfer_state.copy()

event_times = {
    'first_burn': 0.0,
    'first_coast': transfer_time1 / 2,
    'second_burn': transfer_time1,
    'second_coast': transfer_time1 + transfer_time2 / 2,
    'third_burn': transfer_time1 + transfer_time2,
    'orbit_complete': total_transfer_time * 1.2
}

orbit_line, = ax_orbit.plot([], [], 'red', linewidth=2, alpha=0.8, label='Spacecraft')
transfer_orbit_line, = ax_orbit.plot([], [], 'orange', linewidth=2, alpha=0.6, 
                                   linestyle='--', label='Transfer Orbits')

circle1 = plt.Circle((0, 0), params['r1'], fill=False, color='blue', 
                    linestyle='-', alpha=0.7, label='Initial Orbit')
circle2 = plt.Circle((0, 0), params['r2'], fill=False, color='green', 
                    linestyle='-', alpha=0.7, label='Final Orbit')
circle3 = plt.Circle((0, 0), params['r3'], fill=False, color='purple', 
                    linestyle=':', alpha=0.5, label='Intermediate Orbit')

ax_orbit.add_patch(circle1)
ax_orbit.add_patch(circle2)
ax_orbit.add_patch(circle3)

central_body = plt.Circle((0, 0), 0.3, facecolor='yellow', 
                         edgecolor='orange', linewidth=2, zorder=10)
ax_orbit.add_patch(central_body)

spacecraft = plt.Circle((state[0], state[1]), 0.15, facecolor='red', 
                       alpha=0.9, zorder=10, edgecolor='darkred', linewidth=1)
ax_orbit.add_patch(spacecraft)

event_highlight = plt.Circle((state[0], state[1]), 0.4, fill=False, 
                           color='yellow', linewidth=4, alpha=0, zorder=15)
ax_orbit.add_patch(event_highlight)

max_extent = max(params['r2'], params['r3']) * 1.3
ax_orbit.set_xlim(-max_extent, max_extent)
ax_orbit.set_ylim(-max_extent, max_extent)
ax_orbit.set_title('Bielliptic Transfer Orbit Simulation', fontsize=16, fontweight='bold', pad=15)
ax_orbit.set_xlabel('x units', fontsize=12)
ax_orbit.set_ylabel('y units', fontsize=12)
ax_orbit.legend(loc='upper right', fontsize=10)
ax_orbit.grid(True, alpha=0.3)

altitude_line, = ax_altitude.plot([], [], 'purple', linewidth=2)
ax_altitude.set_title('Altitude vs Time', fontsize=9, pad=15)
ax_altitude.set_xlabel('Time', fontsize=9)
ax_altitude.set_ylabel('Distance', fontsize=9)
ax_altitude.grid(True, alpha=0.3)
ax_altitude.tick_params(labelsize=8)

velocity_line, = ax_velocity.plot([], [], 'red', linewidth=2)
ax_velocity.set_title('Velocity vs Time', fontsize=9, pad=15)
ax_velocity.set_xlabel('Time', fontsize=9)
ax_velocity.set_ylabel('Velocity', fontsize=9)
ax_velocity.grid(True, alpha=0.3)
ax_velocity.tick_params(labelsize=8)

energy_line, = ax_energy.plot([], [], 'black', linewidth=2)
ax_energy.set_title('Specific Energy', fontsize=9, pad=15)
ax_energy.set_xlabel('Time', fontsize=9)
ax_energy.set_ylabel('Energy', fontsize=9)
ax_energy.grid(True, alpha=0.3)
ax_energy.tick_params(labelsize=8)

phase_line, = ax_phase.plot([], [], 'blue', linewidth=2)
ax_phase.set_title('Phase Space (r vs v)', fontsize=9, pad=8)
ax_phase.set_xlabel('Distance', fontsize=8)
ax_phase.set_ylabel('Velocity', fontsize=8)
ax_phase.grid(True, alpha=0.3)
ax_phase.tick_params(labelsize=7)

ax_events.set_title('Mission Events', fontsize=9, fontweight='bold', pad=8)
ax_events.set_xlim(0, 1)
ax_events.set_ylim(0, 6)
ax_events.axis('off')

ax_mission.axis('off')
ax_mission.set_title('Mission Parameters', fontsize=12, fontweight='bold', pad=10)

ax_delta_v.axis('off')
ax_delta_v.set_title('ΔV Requirements', fontsize=9, fontweight='bold', pad=8)

plt.subplots_adjust(top=0.9, bottom=0, left=0.06, right=0.96)

slider_width = 0.12
slider_height = 0.015
slider_y_start = 0.08
gap = 0.1

ax_M = plt.axes([0.08, slider_y_start, slider_width, slider_height], facecolor='lightgoldenrodyellow')
ax_r1 = plt.axes([0.08 + slider_width + gap, slider_y_start, slider_width, slider_height], facecolor='lightgoldenrodyellow')
ax_r2 = plt.axes([0.08 + 2*(slider_width + gap), slider_y_start, slider_width, slider_height], facecolor='lightgoldenrodyellow')
ax_phase_slider = plt.axes([0.08 + 3*(slider_width + gap), slider_y_start, slider_width, slider_height], facecolor='lightgoldenrodyellow')

slider_M = Slider(ax_M, 'Mass', 50.0, 200.0, valinit=M_central_init, valstep=5.0)
slider_r1 = Slider(ax_r1, 'R1', 2.0, 8.0, valinit=r1_init, valstep=0.5)
slider_r2 = Slider(ax_r2, 'R2', 8.0, 20.0, valinit=r2_init, valstep=0.5)
slider_phase = Slider(ax_phase_slider, 'R3', 10.0, 25.0, valinit=r3_init, valstep=0.5)  # Changed to R3 for bielliptic

ax_speed = plt.axes([0.075, 0.05, 0.8, 0.015], facecolor='lightcyan')
speed_slider = Slider(ax_speed, 'Speed', 0.1, 15.0, valinit=1.0, valstep=0.1)

ax_reset = plt.axes([0.45, 0.001, 0.1, 0.025])
reset_button = Button(ax_reset, 'Reset')

def reset_simulation():
    global state, current_time, transfer_start_time, simulation_mode
    global delta_v1, delta_v2, delta_v3, transfer_time1, transfer_time2, total_transfer_time
    global a1_transfer, a2_transfer, event_times, v1_circular, v2_circular, r3_actual
    
    params['M_central'] = slider_M.val
    params['r1'] = slider_r1.val
    params['r2'] = slider_r2.val
    params['r3'] = slider_phase.val  # Using phase slider for R3
    params['phase'] = 0.0  # Fixed phase for simplicity
    
    if params['r2'] <= params['r1']:
        params['r2'] = params['r1'] + 1.0
        slider_r2.set_val(params['r2'])
    
    if params['r3'] <= max(params['r1'], params['r2']):
        params['r3'] = max(params['r1'], params['r2']) + 2.0
        slider_phase.set_val(params['r3'])
    
    (transfer_state, delta_v1, delta_v2, delta_v3, 
     transfer_time1, transfer_time2, total_transfer_time,
     a1_transfer, a2_transfer, v1_circular, v2_circular, r3_actual) = setup_bielliptic_transfer(
        params['r1'], params['r2'], params['r3'], params['M_central'], params['phase'])
    
    state = transfer_state.copy()
    current_time = 0.0
    transfer_start_time = 0.0
    simulation_mode = 'transfer1'
    
    event_times = {
        'first_burn': 0.0,
        'first_coast': transfer_time1 / 2,
        'second_burn': transfer_time1,
        'second_coast': transfer_time1 + transfer_time2 / 2,
        'third_burn': transfer_time1 + transfer_time2,
        'orbit_complete': total_transfer_time * 1.2
    }
    
    max_extent = max(params['r2'], params['r3']) * 1.3
    ax_orbit.set_xlim(-max_extent, max_extent)
    ax_orbit.set_ylim(-max_extent, max_extent)
    
    circle1.set_radius(params['r1'])
    circle2.set_radius(params['r2'])
    circle3.set_radius(params['r3'])
    
    time_data.clear()
    x_data.clear()
    y_data.clear()
    altitude_data.clear()
    velocity_data.clear()
    energy_data.clear()
    highlighted_events.clear()

def update_params(val):
    reset_simulation()

slider_M.on_changed(update_params)
slider_r1.on_changed(update_params)
slider_r2.on_changed(update_params)
slider_phase.on_changed(update_params)
reset_button.on_clicked(lambda x: reset_simulation())

def update_mission_display():
    ax_mission.clear()
    ax_mission.axis('off')
    ax_mission.set_title('Mission Parameters', fontsize=12, fontweight='bold', pad=10)
    
    current_stage = "First Burn"
    if current_time > event_times['first_burn'] + 0.5:
        current_stage = "First Transfer"
    if current_time > event_times['first_coast']:
        current_stage = "Approaching Apoapsis"
    if current_time > event_times['second_burn'] - 0.5:
        current_stage = "Second Burn"
    if current_time > event_times['second_burn'] + 0.5:
        current_stage = "Second Transfer"
    if current_time > event_times['second_coast']:
        current_stage = "Approaching Target"
    if current_time > event_times['third_burn'] - 0.5:
        current_stage = "Final Burn"
    if current_time > event_times['third_burn'] + 0.5:
        current_stage = "Final Orbit"
    
    info_text = f"""Central Mass: {params['M_central']:.1f}
Initial Orbit: {params['r1']:.1f}
Final Orbit: {params['r2']:.1f}
Intermediate: {params['r3']:.1f}

Transfer Time 1: {transfer_time1:.2f}
Transfer Time 2: {transfer_time2:.2f}
Total Time: {total_transfer_time:.2f}

ΔV₁: {delta_v1:.3f}
ΔV₂: {delta_v2:.3f}
ΔV₃: {delta_v3:.3f}
Total ΔV: {delta_v1 + delta_v2 + delta_v3:.3f}

Current Time: {current_time:.2f}

CURRENT STAGE:
{current_stage}"""
    
    ax_mission.text(0.02, 0.98, info_text, transform=ax_mission.transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace')

def update_events_display():
    ax_events.clear()
    ax_events.set_xlim(0, 1)
    ax_events.set_ylim(0, 6)
    ax_events.axis('off')
    ax_events.set_title('Mission Events', fontsize=9, fontweight='bold', y=0.8)
    
    events = [
        ("First Burn", event_times['first_burn'], 'red'),
        ("First Coast", event_times['first_coast'], 'orange'),
        ("Second Burn", event_times['second_burn'], 'yellow'),
        ("Second Coast", event_times['second_coast'], 'cyan'),
        ("Third Burn", event_times['third_burn'], 'green'),
        ("Complete", event_times['orbit_complete'], 'blue')
    ]
    
    for i, (name, time, color) in enumerate(events):
        y_pos = 0.75 - i * 0.12
        
        if is_near_event(current_time, time, 0.5):
            ax_events.text(0.02, y_pos, f"► {name}", fontsize=8, fontweight='bold', 
                          color=color, transform=ax_events.transAxes)
        else:
            ax_events.text(0.02, y_pos, f"  {name}", fontsize=7, 
                          color=color, alpha=0.7, transform=ax_events.transAxes)
        
        ax_events.text(0.65, y_pos, f"t={time:.1f}", fontsize=6, 
                      color='gray', transform=ax_events.transAxes)

def update_delta_v_display():
    ax_delta_v.clear()
    ax_delta_v.axis('off')
    ax_delta_v.set_title('ΔV Requirements', fontsize=9, fontweight='bold', pad=8)
    
    current_v = np.sqrt(state[2]**2 + state[3]**2)
    current_r = np.sqrt(state[0]**2 + state[1]**2)
    
    # Compare with Hohmann transfer
    hohmann_dv1 = np.sqrt(G * params['M_central'] * (2/params['r1'] - 2/(params['r1'] + params['r2']))) - v1_circular
    hohmann_dv2 = np.sqrt(G * params['M_central'] / params['r2']) - np.sqrt(G * params['M_central'] * (2/params['r2'] - 2/(params['r1'] + params['r2'])))
    hohmann_total = hohmann_dv1 + hohmann_dv2
    
    delta_v_text = f"""Bielliptic Transfer:
ΔV₁ (r1→r3): {delta_v1:.3f}
ΔV₂ (at r3): {delta_v2:.3f}
ΔV₃ (r3→r2): {delta_v3:.3f}
Total ΔV: {delta_v1 + delta_v2 + delta_v3:.3f}

Hohmann Comparison:
Hohmann ΔV: {hohmann_total:.3f}
Difference: {(delta_v1 + delta_v2 + delta_v3) - hohmann_total:.3f}

Current Velocity: {current_v:.3f}
Current Radius: {current_r:.2f}"""
    
    ax_delta_v.text(0.02, 0.98, delta_v_text, transform=ax_delta_v.transAxes,
                   fontsize=8, verticalalignment='top', fontfamily='monospace')

def animate(frame):
    global state, current_time, simulation_mode, transfer_start_time
    
    dt = base_dt * speed_slider.val
    
    is_event_active = False
    event_color = 'yellow'
    
    for event_name, event_time in event_times.items():
        if is_near_event(current_time, event_time, 0.3):
            is_event_active = True
            if 'burn' in event_name:
                if event_name == 'first_burn':
                    event_color = 'red'
                elif event_name == 'second_burn':
                    event_color = 'yellow'
                elif event_name == 'third_burn':
                    event_color = 'green'
                spacecraft.set_facecolor('orange')
                spacecraft.set_edgecolor('red')
                spacecraft.set_linewidth(3)
            else:
                event_color = 'blue'
                spacecraft.set_facecolor('red')
                spacecraft.set_edgecolor('darkred')
                spacecraft.set_linewidth(1)
            break
    
    if not is_event_active:
        spacecraft.set_facecolor('red')
        spacecraft.set_edgecolor('darkred')
        spacecraft.set_linewidth(1)
    
    event_highlight.set_alpha(0.8 if is_event_active else 0)
    event_highlight.set_color(event_color)
    event_highlight.center = (state[0], state[1])
    
    sol = solve_ivp(spacecraft_dynamics, (current_time, current_time + dt), 
                    state, args=(params['M_central'],),
                    rtol=1e-10, atol=1e-12, method='DOP853')
    
    if sol.success:
        state = sol.y[:, -1]
        current_time += dt
        
        # Handle burns at specific times
        current_r = np.sqrt(state[0]**2 + state[1]**2)
        current_v = np.sqrt(state[2]**2 + state[3]**2)
        
        # Second burn at apoapsis (r3)
        if (simulation_mode == 'transfer1' and 
            abs(current_time - event_times['second_burn']) < 0.1 and
            abs(current_r - params['r3']) < 1.0):
            
            # Apply second burn
            v3_transfer1 = np.sqrt(G * params['M_central'] * (2/params['r3'] - 1/a1_transfer))
            v3_transfer2 = np.sqrt(G * params['M_central'] * (2/params['r3'] - 1/a2_transfer))
            
            if current_v > 0:
                velocity_ratio = v3_transfer2 / v3_transfer1
                state[2] *= velocity_ratio
                state[3] *= velocity_ratio
            
            simulation_mode = 'transfer2'
        
        # Third burn at final orbit
        elif (simulation_mode == 'transfer2' and 
              abs(current_time - event_times['third_burn']) < 0.1 and
              abs(current_r - params['r2']) < 1.0):
            
            # Apply third burn for circularization
            v_circular = np.sqrt(G * params['M_central'] / params['r2'])
            if current_v > 0:
                state[2] = state[2] / current_v * v_circular
                state[3] = state[3] / current_v * v_circular
            
            simulation_mode = 'circular'
        
        energy, h, a, e, r, v = calculate_orbital_elements(state, params['M_central'])
        
        time_data.append(current_time)
        x_data.append(state[0])
        y_data.append(state[1])
        altitude_data.append(r)
        velocity_data.append(v)
        energy_data.append(energy)
        
        if len(time_data) > max_points:
            time_data.pop(0)
            x_data.pop(0)
            y_data.pop(0)
            altitude_data.pop(0)
            velocity_data.pop(0)
            energy_data.pop(0)
        
        spacecraft.center = (state[0], state[1])
        
        orbit_line.set_data(x_data, y_data)
        
        altitude_line.set_data(time_data, altitude_data)
        velocity_line.set_data(time_data, velocity_data)
        energy_line.set_data(time_data, energy_data)
        
        # Add event markers
        for event_name, event_time in event_times.items():
            if len(time_data) > 0 and min(time_data) <= event_time <= max(time_data):
                if 'burn' in event_name:
                    if event_name == 'first_burn':
                        ax_velocity.axvline(x=event_time, color='red', linestyle='--', alpha=0.7)
                    elif event_name == 'second_burn':
                        ax_velocity.axvline(x=event_time, color='yellow', linestyle='--', alpha=0.7)
                    elif event_name == 'third_burn':
                        ax_velocity.axvline(x=event_time, color='green', linestyle='--', alpha=0.7)
        
        phase_line.set_data(altitude_data, velocity_data)
        
        for ax in [ax_altitude, ax_velocity, ax_energy, ax_phase]:
            ax.relim()
            ax.autoscale_view()
        
        update_mission_display()
        update_events_display()
        update_delta_v_display()
    
    return (orbit_line, spacecraft, event_highlight, altitude_line, velocity_line, energy_line, phase_line)

reset_simulation()

ani = FuncAnimation(fig, animate, frames=100000, interval=50, blit=False, repeat=True)

plt.show()
