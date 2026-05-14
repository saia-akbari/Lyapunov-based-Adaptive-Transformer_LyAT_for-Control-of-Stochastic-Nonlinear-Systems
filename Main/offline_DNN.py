import torch
import torch.nn as nn
import numpy as np
import math
import json
import os

#============================================================
# Author: Saiedeh Akbari
#============================================================


# ================== Dynamical System ====================== #
class Dynamics:
    """A six-dimensional nonlinear stochastic system dynamics."""

    @staticmethod
    def drift_vector(x):
        """Compute f(x) - the drift vector."""
        x1, x2, x3, x4, x5, x6 = x

        f1 = 5 * torch.tanh(50 * x1) * x5**2 + torch.cos(x4)
        f2 = torch.cos(20 * x3) + 2 * torch.sin(x1 * x2) * torch.sin(x4 * x5)
        f3 = 10 * torch.exp(-25 * x4**2) * x3 - 0.1 * x3**3
        f4 = 2 * torch.sin(15 * (x1 * x5 - x2 * x3))
        f5 = -x1 * x5 + 5 * torch.tanh(20 * (x2 - x4))
        f6 = -0.1 * x6 + torch.sin(x1 * x2)

        return torch.stack([f1, f2, f3, f4, f5, f6])

    @staticmethod
    def control_effectiveness():
        return torch.eye(6)

    @staticmethod
    def diffusion_matrix(x):
        x1, x2, x3, x4, x5, x6 = x

        g2 = torch.zeros(6, 2)

        g2[0, 0] = x1 * torch.cos(x2)
        g2[0, 1] = 1 - x3 * torch.cos(x4)
        g2[1, 0] = x3 * x5
        g2[1, 1] = x4**2 * torch.sin(x2)**2
        g2[2, 0] = x1**2
        g2[2, 1] = x3 * torch.cos(x1 * x2)
        g2[3, 0] = (x1 + x2)**3 - torch.sin(x3)
        g2[3, 1] = 1 - x3**2
        g2[4, 0] = x2 * torch.sin(x3)**2
        g2[4, 1] = -x5 + x1 * x4**2
        g2[5, 0] = torch.cos(x6)
        g2[5, 1] = x6**2 + x1 * x2

        return g2

    @staticmethod
    def covariance_matrix(t):
        Sigma = torch.zeros(2, 2)
        Sigma[0, 0] = torch.sin(t)**2
        Sigma[1, 1] = torch.exp(-t)
        return Sigma

    @staticmethod
    def desired_trajectory(t):
        height = 2.5  # meters
        omega = 0.15  # rad/s

        z_tilt = 0.0
        a = 7.5  # Half-width of the long side (x-direction)
        b = 3.0  # Half-width of the short side (y-direction)

        # Position (figure 8 with major axis along x)
        xd1 = a * torch.sin(omega * t)
        xd2 = b * torch.sin(2.0 * omega * t)
        xd3 = torch.tensor(height, dtype=torch.float32) + (z_tilt / 2) * torch.sin(omega * t)
        # Velocity
        xd4 = a * omega * torch.cos(omega * t)
        xd5 = 2 * b * omega * torch.cos(2.0 * omega * t)
        xd6 = (z_tilt / 2) * omega * torch.cos(omega * t)

        xd4_dot = -a * omega**2 * torch.sin(omega * t)
        xd5_dot = -4 * b * omega**2 * torch.sin(2.0 * omega * t)
        xd6_dot = -(z_tilt / 2) * omega**2 * torch.sin(omega * t)

        xd = torch.stack([xd1, xd2, xd3, xd4, xd5, xd6])
        xd_dot = torch.stack([xd4, xd5, xd6, xd4_dot, xd5_dot, xd6_dot])

        return xd, xd_dot


# ==================== Offline DNN ======================== #
class UncertaintyDNN(nn.Module):
    """Standard feedforward DNN to approximate drift uncertainty f(x)."""

    def __init__(self, n_states=6, hidden_dim=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(n_states, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, n_states))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        torch.manual_seed(0)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== Data Collection ==================== #
def collect_training_data(config):
    """Simulate under a baseline controller (Phi=0) and collect (x, f(x)) pairs."""
    n_states = config['n_states']
    dt = config['dt']
    ke = config['ke']
    T_collect = config.get('T_collect', config['T_final'] / 4.0)

    time_steps = int(T_collect / dt)
    x = torch.tensor([0.0, -1.0, 3.0, -3.0, 3.0, 0.0], dtype=torch.float32)

    X_data = []
    Y_data = []

    print("Phase 1: Collecting training data...")
    for step in range(1, time_steps):
        t = step * dt
        t_tensor = torch.tensor(t, dtype=torch.float32)

        xd, xd_dot = Dynamics.desired_trajectory(t_tensor)
        e = x - xd

        # Baseline controller: Phi = 0 (no uncertainty compensation)
        g1 = Dynamics.control_effectiveness()
        g1_inv = torch.inverse(g1)
        u_baseline = g1_inv @ (xd_dot - ke * e)

        # True drift f(x) is the uncertainty the DNN must approximate
        f = Dynamics.drift_vector(x)

        X_data.append(x.detach().clone())
        Y_data.append(f.detach().clone())

        g2 = Dynamics.diffusion_matrix(x)
        Sigma = Dynamics.covariance_matrix(t_tensor)
        dw = torch.randn(2) * math.sqrt(dt)
        x = x + (f + g1 @ u_baseline) * dt + g2 @ Sigma @ dw

        if step % 1000 == 0:
            progress = 100 * step / time_steps
            print(f"\r  Collection: {progress:.1f}%", end="", flush=True)

    print(f"\n  Collected {len(X_data)} samples.")
    X = torch.stack(X_data)
    Y = torch.stack(Y_data)
    return X, Y


# ==================== Offline Training =================== #
def train_dnn(dnn, X, Y, config):
    """Train DNN offline using Adam optimizer, then freeze weights for inference."""
    epochs = config.get('dnn_epochs', 300)
    lr = config.get('dnn_lr', 1e-3)
    batch_size = config.get('dnn_batch_size', 256)

    optimizer = torch.optim.Adam(dnn.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    N = X.shape[0]

    print("\nPhase 2: Training offline DNN...")
    dnn.train()
    for epoch in range(epochs):
        perm = torch.randperm(N)
        X_shuf = X[perm]
        Y_shuf = Y[perm]

        total_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            x_batch = X_shuf[i:i + batch_size]
            y_batch = Y_shuf[i:i + batch_size]
            optimizer.zero_grad()
            y_pred = dnn(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch + 1}/{epochs}]  Loss: {total_loss / n_batches:.6f}")

    # Freeze weights: no further adaptation at inference time
    dnn.eval()
    for param in dnn.parameters():
        param.requires_grad_(False)

    print(f"  DNN training complete. Parameters: {dnn.count_parameters()}")
    return dnn


# ==================== Controller ========================= #
class OfflineDNN_Controller:
    """Same control law as LyAT_Controller but Phi is provided by a frozen, offline-trained DNN."""

    def __init__(self, config, dnn):
        self.config = config
        self.n_states = config['n_states']
        self.ke = config['ke']
        self.dt = config['dt']
        self.dnn = dnn  # weights frozen after offline training

    def compute_control(self, x, t):
        xd, xd_dot = Dynamics.desired_trajectory(torch.tensor(t, dtype=torch.float32))
        e = x - xd

        with torch.no_grad():
            Phi = self.dnn(x.unsqueeze(0)).squeeze(0)

        g1 = Dynamics.control_effectiveness()
        g1_inv = torch.inverse(g1)
        u = g1_inv @ (xd_dot - self.ke * e - Phi)

        return u.detach(), Phi.detach()


# ==================== Simulation ========================= #
class Simulation:
    def __init__(self, config_path='config_offline_dnn.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.n_states = self.config['n_states']
        self.dt = self.config['dt']
        self.T_final = self.config['T_final']
        self.time_steps = int(self.T_final / self.dt)

        self.x = torch.zeros((self.n_states, self.time_steps))
        self.x[:, 0] = torch.tensor([0.0, -1.0, 3.0, -3.0, 3.0, 0.0], dtype=torch.float32)

    def update_state(self, step, u, t):
        x_current = self.x[:, step - 1]

        f = Dynamics.drift_vector(x_current)
        g1 = Dynamics.control_effectiveness()
        g2 = Dynamics.diffusion_matrix(x_current)
        Sigma = Dynamics.covariance_matrix(torch.tensor(t, dtype=torch.float32))
        dw = torch.randn(2) * math.sqrt(torch.tensor(self.dt))

        self.x[:, step] = self.x[:, step - 1] + (f + g1 @ u) * self.dt + g2 @ Sigma @ dw

    def run(self, dnn):
        controller = OfflineDNN_Controller(self.config, dnn)
        tracking_errors = []
        control_inputs = []
        parameter_norms = []

        print("\nPhase 3: Running controlled simulation...")
        for step in range(1, self.time_steps):
            t = step * self.dt

            if step % 100 == 0:
                progress = 100 * step / self.time_steps
                theta = torch.cat([p.view(-1) for p in controller.dnn.parameters()])
                param_norm = torch.norm(theta).item()
                parameter_norms.append(param_norm)
                print(f"\rProgress: {progress:.1f}% | ||θ||: {param_norm:.2f}", end="", flush=True)

            x = self.x[:, step - 1]
            xd, xd_dot = Dynamics.desired_trajectory(torch.tensor(t, dtype=torch.float32))
            u, Phi = controller.compute_control(x, t)
            self.update_state(step, u, t)

            e = self.x[:, step] - xd
            error_norm = torch.norm(e).item()
            tracking_errors.append(error_norm)
            control_inputs.append(torch.norm(u).item())

        print(f"\n\nSimulation complete!")
        print(f"RMS tracking error: {np.sqrt(np.mean(np.square(tracking_errors))):.6f}")
        print(f"RMS control input:  {np.sqrt(np.mean(np.square(control_inputs))):.6f}")

        self.results = {
            'x': self.x,
            'tracking_errors': tracking_errors,
            'control_inputs': control_inputs,
            'parameter_norms': parameter_norms,
            'time': np.arange(1, self.time_steps) * self.dt
        }

        return self.results

    def save_results(self, filename='offline_dnn_results.json'):
        results_dict = {
            'tracking_errors': [float(e) for e in self.results['tracking_errors']],
            'control_inputs': [float(u) for u in self.results['control_inputs']],
            'time': [float(t) for t in self.results['time']],
            'rms_error': float(np.sqrt(np.mean(np.square(self.results['tracking_errors']))))
        }

        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=4)

        print(f"\nResults saved to {filename}")

    def plot_results(self):
        import matplotlib.pyplot as plt

        time = self.results['time']
        tracking_errors = self.results['tracking_errors']

        plt.figure(figsize=(10, 6))
        plt.plot(time, tracking_errors, 'b-', linewidth=2)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Tracking Error ||e(t)||', fontsize=14)
        plt.title('Offline DNN Controller: Tracking Error', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('offline_dnn_tracking_error.png', dpi=300)
        print("\nPlot saved as 'offline_dnn_tracking_error.png'")
        plt.show()


# ==================== Main =============================== #
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    with open('lyapunov_adaptive_transformer/lyapunov_adaptive_transformer/config.json', 'r') as config_file:
        config = json.load(config_file)

    with open('config_offline_dnn.json', 'w') as f:
        json.dump(config, f, indent=4)
    print("Config file created: config_offline_dnn.json")

    # Phase 1: collect training data by simulating with baseline controller
    X, Y = collect_training_data(config)

    # Phase 2: train DNN offline with Adam
    dnn = UncertaintyDNN(
        n_states=config['n_states'],
        hidden_dim=config['dnn_hidden_dim'],
        n_layers=config['dnn_n_layers']
    )
    dnn = train_dnn(dnn, X, Y, config)

    # Phase 3: freeze weights, run full controlled simulation
    sim = Simulation('config_offline_dnn.json')
    results = sim.run(dnn)

    sim.save_results('offline_dnn_results.json')
    sim.plot_results()

    print("\n" + "=" * 70)
    print("SIMULATION FINISHED!")
    print("=" * 70)


if __name__ == "__main__":
    main()