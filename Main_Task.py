"""
Sail Drone Simulation - Hardcoded Control
Author: Louise Mitchell
Date: Nov-2025
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation

# Fixing the working directory for my laptop.
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Running from: {os.getcwd()}")

from saildrone_hydro import get_vals

def get_control_inputs(t):
    """
    Get sail and rudder angles for the basic sailing course.
    
    Parameters
    ----------
    t : float
        Current time in seconds
        
    Returns
    ----------
    beta_sail : float
        Sail angle in radians, relative to boat centreline
    beta_rudder : float 
        Rudder angle in radians, relative to boat centreline
    """
    if t < 60:
        # Leg A
        beta_sail = np.radians(-45)
        beta_rudder = np.radians(0)
    elif 60 <= t < 65:
        # Leg B
        beta_sail = np.radians(-22.5)
        beta_rudder = np.radians(2.1)
    else:
        # Leg C
        beta_sail = np.radians(-22.5)
        beta_rudder = np.radians(0)
    
    return beta_sail, beta_rudder

def calculate_aerodynamic_forces(vx, vy, theta, beta_sail):
    """
    Calculate aerodynamic forces from the sail.
    
    Parameters
    ----------
    vx, vy : float
        Vessel x and y velocity components (m/s)
    theta : float  
        Vessel heading (radians)
    beta_sail : float
        Sail angle relative to vessel (radians)
        
    Returns
    -------
    F_aero_x, F_aero_y : float
        Aerodynamic force components in global coordinates (N)
    """
    # Apparent wind calculation
    v_wind_true = np.array([-6.7, 0])  # Wind from East at 6.7 m/s as given in brief
    v_boat = np.array([vx, vy])
    v_wind_app = v_wind_true - v_boat
    v_app_mod = np.linalg.norm(v_wind_app)  # Apparent wind speed magnitude
    
    # Check for zero wind to avoid division issues
    if v_app_mod < 0.001:
        return 0.0, 0.0
    
    # Calculate apparent wind direction (angle)
    wind_dir = np.arctan2(v_wind_app[1], v_wind_app[0])
    
    # Sail angle of attack
    sail_orient = theta + beta_sail
    alpha = sail_orient - wind_dir  
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
    
    # Aerodynamic coefficients
    A = 15                                                   # Sail area (m²)
    rho = 1.225                                              # Air density (kg/m³)
    C_D = 1 - np.cos(2 * alpha)                              # Drag coefficient as given in brief
    C_L = 1.5 * np.sin(2 * alpha + 0.5 * np.sin(2 * alpha))  # Lift coefficient as given in brief
    
    # Aerodynamic calculations
    q = 0.5 * rho * v_app_mod**2
    drag_force = q * A * C_D
    lift_force = q * A * C_L
    
    # Convert to global coordinates 
    drag_dir = wind_dir + np.pi    # Drag opposite to wind direction
    lift_dir = wind_dir - np.pi/2  # Lift perpendicular to wind
    
    F_drag_x = drag_force * np.cos(drag_dir)
    F_drag_y = drag_force * np.sin(drag_dir)
    F_lift_x = lift_force * np.cos(lift_dir) 
    F_lift_y = lift_force * np.sin(lift_dir)
    
    F_aero_x = F_lift_x + F_drag_x
    F_aero_y = F_lift_y + F_drag_y
    
    return F_aero_x, F_aero_y

def state_deriv_saildrone(t, z):
    """
    Compute the derivative of the state vector for the sail drone system.
    
    Parameters
    ----------
    t : float64
        Time in seconds.
    z : numpy.ndarray (6 elements)
        State vector [x, y, theta, vx, vy, omega]
        
    Returns
    -------
    dz : numpy.ndarray (6 elements)
        Derivative of the state vector [dx/dt, dy/dt, dtheta/dt, dvx/dt, dvy/dt, domega/dt]
    """
    # System parameters
    M = 2500      # mass (kg)
    I = 10000     # rotational inertia (kg·m²)
    
    # Unpack state vector
    x, y, theta, vx, vy, omega = z
    
    # Trivial derivatives
    dz1 = vx      # dx/dt
    dz2 = vy      # dy/dt
    dz3 = omega   # dtheta/dt
    
    # Get control inputs - hardcoded for course
    beta_sail, beta_rudder = get_control_inputs(t)
    
    # Calculate aerodynamic forces
    F_aero_x, F_aero_y = calculate_aerodynamic_forces(vx, vy, theta, beta_sail)
    
    # Get hydrodynamic forces    
    velocity_vector = np.array([vx, vy])
    F_hydro, torque_hydro = get_vals(velocity_vector, theta, omega, beta_rudder)

    
    # Calculate aerodynamic torque from sail position (100mm aft)
    sail_position_aft = 0.1  # meters
    sail_global_x = -sail_position_aft * np.cos(theta)
    sail_global_y = -sail_position_aft * np.sin(theta)
    torque_aero = sail_global_x * F_aero_y - sail_global_y * F_aero_x
    
    # Total forces and torques
    F_total_x = F_aero_x + F_hydro[0]
    F_total_y = F_aero_y + F_hydro[1]
    torque_total = torque_aero + torque_hydro
    
    # Accelerations (the dynamic derivatives)
    dz4 = F_total_x / M  # dvx/dt
    dz5 = F_total_y / M  # dvy/dt
    dz6 = torque_total / I  # domega/dt
    
    return np.array([dz1, dz2, dz3, dz4, dz5, dz6])

def step_rk(state_deriv, t, dt, z):
    """
    Apply fourth-order Runge-Kutta method for one time step.
    """
    # Fourth-order Runge-Kutta method
    k1 = dt * state_deriv(t, z)
    k2 = dt * state_deriv(t + dt/2, z + k1/2)
    k3 = dt * state_deriv(t + dt/2, z + k2/2)
    k4 = dt * state_deriv(t + dt, z + k3)
    
    # Weighted average of all slopes
    znext = z + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return znext

def solve_ivp(state_deriv, t0, tmax, dt, z0):
    """
    Solve an initial value problem (IVP) for an ordinary differential equation (ODE)
    """
    # Set initial conditions
    t = np.array([t0])
    z = z0.reshape(-1, 1)  # Ensure z is 2D array
    
    # Continue stepping until the end time is exceeded
    n = 0
    while t[n] < tmax:
        # Increment by one time step and append to the time axis
        t = np.append(t, t[-1] + dt)
        
        # Using Runge-Kutta method
        znext = step_rk(state_deriv, t[n], dt, z[:, n])
        
        # Append to the solution
        znext = znext.reshape(-1, 1)
        z = np.append(z, znext, axis=1)
        
        n = n + 1
    
    return t, z

def animate_trajectory(t, z):
    """
    Create an animation of the sail drone trajectory.
    """
    
    x, y, theta, vx, vy, omega = z
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(np.min(x)-10, np.max(x)+10)
    ax.set_ylim(np.min(y)-10, np.max(y)+10)
    ax.set_xlabel('East Position (m)')
    ax.set_ylabel('North Position (m)')
    ax.set_title('Sail Drone Navigation - Real Time')
    ax.grid(True, alpha=0.3)
    
    # Plot full trajectory
    trajectory, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)
    drone, = ax.plot([], [], 'ro', markersize=8)
    heading, = ax.plot([], [], 'r-', linewidth=2)
    
    def update(frame):
        trajectory.set_data(x[:frame], y[:frame])
        drone.set_data(x[frame], y[frame])
        
        # Show heading direction
        arrow_length = 5
        heading_x = [x[frame], x[frame] + arrow_length * np.cos(theta[frame])]
        heading_y = [y[frame], y[frame] + arrow_length * np.sin(theta[frame])]
        heading.set_data(heading_x, heading_y)
        
        return trajectory, drone, heading
    
    anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True, repeat=False)
    plt.show()
    
    return anim

def simulate_saildrone():
    """
    Run the complete sail drone simulation and plot results.
    """
    # Initial conditions from assignment
    z0 = np.array([
        0,           # x = 0 m
        0,           # y = 0 m  
        np.pi/2,     # theta = 90° (facing North)
        0,           # vx = 0 m/s
        2.9,         # vy = 2.9 m/s (Northward)
        0            # omega = 0 rad/s
    ])
    
    # Time parameters
    t0 = 0
    tmax = 120      # 2 minutes simulation
    dt = 0.1        # Time step
    
    # Solve the ODEs
    t, z = solve_ivp(state_deriv_saildrone, t0, tmax, dt, z0)
    
    # Extract results
    x_traj = z[0, :]    # x position over time
    y_traj = z[1, :]    # y position over time
    theta_traj = z[2, :] # heading over time
    vx_traj = z[3, :]   # x-velocity over time
    vy_traj = z[4, :]   # y-velocity over time
    omega_traj = z[5, :] # angular velocity over time

    # Draw trajectory
    animate_trajectory(t,z)

    print(f"Final position: ({x_traj[-1]:.1f}, {y_traj[-1]:.1f}) m")
    print(f"Final heading: {np.degrees(theta_traj[-1]):.1f}°")
    print(f"Final speed: {speed[-1]:.1f} m/s") 

# Run the simulation
if __name__ == '__main__':
    simulate_saildrone()