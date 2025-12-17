# Lyapunov-based-Adaptive-Transformer_LyAT_for-Control-of-Stochastic-Nonlinear-Systems
Implementation of LyAT: A Lyapunov-based Adaptive Transformer for stochastic nonlinear control. Features real-time weight adaptation without offline training and provable convergence guarantees (UUB-p). Includes quadrotor experimental validation code.

## Required Software
- Ubuntu 22.04, ROS2 Humble
- PX4-Autopilot 1.15
- Mavros
- Gazebo Fortress
- QGroundControl

## Configuration
{
    "n_states": 6,              # State dimension
    "window_size": 20,          # Historical window size (τ)
    "T_final": 240.0,           # Experiment duration (seconds)
    "dt": 0.005,                # Time step
    "ke": 0.8,                  # Control gain
    "gamma": 0.02,              # Learning rate (Γ)
    "sigma": 0.000001,          # Forgetting factor
    "theta_bar": 10.0,          # Parameter bound
    "num_encoder_layers": 1,    # Number of encoder layers (N)
    "num_decoder_layers": 1,    # Number of decoder layers (N)
    "num_heads": 3,             # Multi-head attention heads (H)
    "d_ff": 5,                  # Feedforward dimension
    "VEL_MAX": 1.8              # Maximum velocity (m/s)
}

## Hardware Deployment
#### For Simulation:
```bash
ros2 launch lyapunov_adaptive_transformer astro_sim_launch.py
```

#### For Hardware (Freefly Astro):
```bash
ros2 launch lyapunov_adaptive_transformer astro2_launch.py
```
### Visualization
Generate plots from saved data:
```bash
python src/plotter.py
```
This will generate:
- Tracking error norm over time
- 3D trajectory comparison (agent vs. target)
- Control input signals (X, Y, Z channels)
- Parameter evolution (θ weights)

## Experimental Results

### Quadrotor Tracking Performance

The LyAT controller was validated on a **Freefly Astro quadrotor** at the University of Florida's Autonomy Park, tracking a figure-8 trajectory:

| Metric | Value |
|--------|-------|
| **Flight Duration** | 240 seconds |
| **RMS Tracking Error** | 0.2175 meters |
| **Trajectory** | Figure-8 (7.5m × 3m) |
| **Altitude** | 2.5 meters |
| **Control Update Rate** | ~20 Hz |

**Key Observations:**
- Rapid convergence from initial conditions (~10 seconds)
- Bounded tracking error throughout the entire flight
- Smooth control signals without chattering
- Real-time adaptation without offline training

## Hardware Setup

### Freefly Astro Quadrotor Specifications
- **Diameter:** 930 mm (1407 mm with propellers)
- **Motors:** Freefly 7010 with 21×7" carbon fiber props
- **Max Takeoff Weight:** 8700 g
- **Flight Controller:** Freefly Skynode (Auterion PX4)
- **Sensors:** GPS (L1/L2), optical flow, lidar, barometer
- **State Estimation:** PX4 EKF2 fusion algorithm

### Communication
- **Protocol:** MAVLink via MAVROS
- **Command Type:** Velocity setpoints
- **Safety:** Maximum velocity saturation at 1.8 m/s

## Authors & Contributors
**Saiedeh Akbari** (akbaris@ufl.edu) - Lead author, core LyAT.py algorithm development and theoretical analysis, LyAT deployment and tuning
**Xuehui Shen** - Co-author, assisted with the theoretical analysis
**Wenqian Xue** - Co-author, assisted with the theoretical analysis
**Jordan Insinger** - Co-author, hardware setup, experimental testbed configuration, ROS setup, lyat_node.py development, and flight operation
**Warren Dixon** - Principal Investigator, supervision, and theoretical analysis

| Component | Primary Developer(s) |
|-----------|---------------------|
| LyAT Core Algorithm (`LyAT.py`) | Saiedeh Akbari |
| ROS2 Integration (`lyat_node.py`) | Jordan Insinger |
| Launch Files | Jordan Insinger |
| Data Management & Plotting | Jordan Insinger, Saiedeh Akbari |
| Hardware Configuration | Jordan Insinger |
| Flight Testing & Data Collection | Jordan Insinger, Saiedeh Akbari |
| Theoretical Framework | Saiedeh Akbari, Xuehui Shen, Wenqian Xue, Warren Dixon |
