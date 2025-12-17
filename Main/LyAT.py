import torch
import torch.nn as nn
import numpy as np
import math
# import matplotlib.pyplot as plt
import json
import os
import rclpy

#============================================================
Author: Saiedeh Akbari
#============================================================

# ================== Dynamical System ====================== #
class Dynamics:
    """ a five dimensional nonlinear stochastic system dynamics"""

    @staticmethod
    def drift_vector (x):
        """ compute f(x) - the drift vector """
        x1, x2, x3, x4, x5 = x

        f1 = 5 * torch.tanh(50 * x1) * x5**2 + torch.cos(x4)
        f2 = torch.cos(20 * x3) + 2 * torch.sin(x1 * x2) * torch.sin(x4 * x5)
        f3 = 10 * torch.exp(-25 * x4**2) * x3 - 0.1 * x3**3
        f4 = 2 * torch.sin(15 * (x1 * x5 - x2 * x3))
        f5 = -x1 * x5 + 5 * torch.tanh(20 * (x2 - x4))

        return torch.stack ([f1, f2, f3, f4, f5])
    
    @staticmethod
    def control_effectiveness ():
        return torch.eye(6)
    
    @staticmethod
    def diffusion_matrix (x):
        x1, x2, x3, x4, x5, x6 = x

        g2 = torch.zeros(6, 2)

        g2[0,0] = x1 * torch.cos(x2)
        g2[0,1] = 1 - x3 * torch.cos(x4)
        g2[1,0] = x3 * x5
        g2[1,1] = x4 ** 2 * torch.sin(x2) ** 2
        g2[2,0] = x1 ** 2
        g2[2,1] = x3 * torch.cos(x1 * x2)
        g2[3,0] = (x1 + x2) ** 3 - torch.sin(x3)
        g2[3,1] = 1 - x3 ** 2
        g2[4,0] = x2 * torch.sin(x3) ** 2
        g2[4,1] = - x5 + x1 * x4 ** 2
        g2[5,0] = torch.cos(x6)
        g2[5,1] = x6 ** 2 + x1 * x2


        return g2
    
    @staticmethod
    def covariance_matrix (t):
        Sigma = torch.zeros(2, 2)
        Sigma[0, 0] = torch.sin(t)**2
        Sigma[1, 1] = torch.exp(-t)
        return Sigma
    
    @staticmethod
    def desired_trajectory(t):
        height = 2.5  # meters
        omega = 0.15  # rad/s
        r = 2.5
        
        # # desired position (figure 8)
        # xd1 = a * torch.sin(omega * t)
        # xd2 = b * torch.sin(2 * omega * t)
        # xd3 = torch.tensor(height, dtype=torch.float32)  # Convert to tensor
        
        # # desired velocity
        # xd4 = a * omega * torch.cos(omega * t)
        # xd5 = 2 * b * omega * torch.cos(2 * omega * t)
        # xd6 = torch.tensor(0.0, dtype=torch.float32)  # Convert to tensor
        
        # # desired acceleration
        # xd4_dot = -a * omega**2 * torch.sin(omega * t)
        # xd5_dot = -4 * b * omega**2 * torch.sin(2 * omega * t)
        # xd6_dot = torch.tensor(0.0, dtype=torch.float32)  # Convert to tensor

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


#================= Transformer Architecture ===============================#

class PositionalEncoding (nn.Module):
    def __init__ (self, d_model, max_len = 100):
        super().__init__()
        self.d_model = d_model
        PE = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        PE [:, 0::2] = torch.sin (position * div_term[:d_model//2 + d_model%2])
        PE [:, 1::2] = torch.cos (position * div_term[:d_model//2])

        self.register_buffer('PE', PE)


    def forward (self, x):
        sequence_length = x.size(1)
        return x + self.PE[:sequence_length, :].unsqueeze(0)
    

class MultiHeadAttention (nn.Module):
    def __init__(self, d_model_query, d_model_kv, num_heads, is_masked = False, gamma = 1.0, beta = 0.0):
        super().__init__()
        assert d_model_query % num_heads == 0

        self.d_model_query = d_model_query
        self.d_model_kv = d_model_kv
        self.num_heads = num_heads
        self.d_k = d_model_query // num_heads
        self.is_masked = is_masked

        self.weights_query = nn.ModuleList ([nn.Linear(d_model_query, self.d_k, bias = False) for _ in range(num_heads)])
        self.weights_key = nn.ModuleList ([nn.Linear(d_model_kv, self.d_k, bias = False) for _ in range(num_heads)])
        self.weights_value = nn.ModuleList ([nn.Linear(d_model_kv, self.d_k, bias = False) for _ in range(num_heads)])

        # This is equivalant to mathcal{W}_\ell^n in the paper
        self.weights_HeadProjection = nn.Linear(d_model_query, d_model_query, bias = False)

        self.register_buffer('gamma', torch.ones(d_model_query) * gamma)
        self.register_buffer('beta', torch.ones(d_model_query) * beta)


    def forward (self, query_input, key_value_input):
        batch_size = query_input.size (0)
        sequence_length = query_input.size (1)
        residual = query_input

        head_outputs = []
        
        for h in range (self.num_heads):
            Q = self.weights_query[h] (query_input)
            K = self.weights_key[h] (key_value_input)
            V = self.weights_value[h] (key_value_input)
        
            scores = torch.matmul (Q, K.transpose (-2, -1)) / math.sqrt (self.d_k)
        
            if self.is_masked:
                mask = torch.triu (torch.ones (sequence_length, sequence_length), diagonal=1).bool()
                mask = mask.to (scores.device)
                scores = scores.masked_fill (mask.unsqueeze(0), float('-inf'))

            attention_weights = torch.softmax (scores, dim = -1)
            head_output = torch.matmul (attention_weights, V)
            head_outputs.append (head_output)
        
        multi_head_output = torch.cat (head_outputs, dim = -1)
        output = self.weights_HeadProjection (multi_head_output)
        output = output + residual
        output = self.layer_norm (output)

        return output
    
    def layer_norm (self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + 1e-8) + self.beta
    

class FeedForward (nn.Module):
    def __init__ (self, d_model, d_ff, gamma = 1.0, beta = 0.0):
        super().__init__()
        self.W_f1 = nn.Linear (d_model, d_ff, bias = False)
        self.W_f2 = nn.Linear (d_ff, d_model, bias = False)

        self.register_buffer('gamma', torch.ones(d_model) * gamma)
        self.register_buffer('beta', torch.ones(d_model) * beta)

    def forward (self, x):
        residual = x
        x = self.W_f2 (torch.relu (self.W_f1 (x)))
        x = x + residual
        x = self.layer_norm (x)

        return x
    
    def layer_norm (self, x):
        mean = x.mean (dim = -1, keepdim = True)
        std = x.std (dim = -1, keepdim = True)
        return self.gamma * (x - mean) / (std + 1e-8) + self.beta
    
class EncoderLayer (nn.Module):
    def __init__ (self, d_model, num_heads, d_ff, gamma_attn = 1.0, beta_attn = 0.0, gamma_ff = 1.0, beta_ff = 0.0):
        super().__init__()
        self.self_attention = MultiHeadAttention (d_model, d_model, num_heads, is_masked = False, gamma = gamma_attn, beta = beta_attn)
        self.feedforward = FeedForward (d_model, d_ff, gamma = gamma_ff, beta = beta_ff)

    def forward (self, x):
        x = self.self_attention (x, x)
        x = self.feedforward (x)
        return x   
class DecoderLayer (nn.Module):
    def __init__ (self, d_model_enc, d_model_dec, num_heads, d_ff, gamma_self = 1.0, beta_self = 0.0, gamma_cross = 1.0, beta_cross = 0.0, gamma_ff = 1.0, beta_ff = 0.0):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention (d_model_dec, d_model_dec, num_heads, is_masked = True, gamma = gamma_self, beta = beta_self)
        self.cross_attention = MultiHeadAttention (d_model_dec, d_model_enc, num_heads, is_masked = False, gamma = gamma_cross, beta = beta_cross)
        self.feedforward = FeedForward (d_model_dec, d_ff, gamma = gamma_ff, beta = beta_ff)
    
    
    def forward (self, x, encoder_output):
        # masked self atention
        x = self.masked_self_attention (x, x)

        # cross attention 
        x = self.cross_attention (x, encoder_output)

        # feedforward
        x = self.feedforward (x)
        
        return x
    
    # ==================== Lyapunov-Based Adaptive Transformer ==============================##

class LyAT (nn.Module):
    def __init__ (self, n_states = 6, window_size = 10, num_encoder_layers = 2, num_decoder_layers = 2, num_heads = 6, d_ff = 128, 
                  # Encoder:
                  gamma_encoder_attn = 1.0, beta_encoder_attn = 0.0,
                  gamma_encoder_ff = 1.0, beta_encoder_ff = 0.0,
                  # Decoder:
                  gamma_decoder_self = 1.0, beta_decoder_self = 0.0,
                  gamma_decoder_cross = 1.0, beta_decoder_cross = 0.0,
                  gamma_decoder_ff = 1.0, beta_decoder_ff = 0.0):
        super().__init__()

        self.n_states = n_states
        self.window_size = window_size
        self.d_model_encoder = 3 * n_states
        self.d_model_decoder = n_states

        self.pos_encoding_encoder = PositionalEncoding (self.d_model_encoder, window_size)
        self.pos_encoding_decoder = PositionalEncoding (self.d_model_decoder, window_size)
        
        self.encoder_layers = nn.ModuleList ([
            EncoderLayer (self.d_model_encoder, num_heads, d_ff,
                          gamma_attn = gamma_encoder_attn, beta_attn = beta_encoder_attn,
                          gamma_ff = gamma_encoder_ff, beta_ff = beta_encoder_ff)
            for _ in range (num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList ([
            DecoderLayer (self.d_model_encoder, self.d_model_decoder, num_heads, d_ff,
                          gamma_self = gamma_decoder_self, beta_self = beta_decoder_self,
                          gamma_cross = gamma_decoder_cross, beta_cross = beta_decoder_cross,
                          gamma_ff = gamma_decoder_ff, beta_ff = beta_decoder_ff)
            for _ in range (num_decoder_layers)
        ])

        self.output_projection = nn.Linear (self.d_model_decoder * window_size, n_states, bias = False)
        self._initialize_weights()

    def _initialize_weights (self):
        torch.manual_seed (0)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)

    def forward (self, zeta_encoder, Phi_history):
        batch_size = zeta_encoder.size(0)

        encoder_input = self.pos_encoding_encoder(zeta_encoder)
        decoder_input = self.pos_encoding_decoder(Phi_history)

        encoder_output = encoder_input
        for layer in self.encoder_layers:
            encoder_output = layer (encoder_output)
        
        decoder_output = decoder_input
        for layer in self.decoder_layers:
            decoder_output = layer (decoder_output, encoder_output)

        decoder_flat = decoder_output.reshape (batch_size, -1)
        Phi = self.output_projection (decoder_flat)

        return Phi
    
    def count_parameters (self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== Controller ======================== #

def smooth_projection (Lambda, theta, theta_bar):

    f = torch.norm (theta) ** 2 - theta_bar ** 2
    del_f = 2 * theta 
    projection_scale = torch.outer (del_f, del_f) / torch.norm (del_f) ** 2

    if (f < 0) or (torch.dot(del_f, Lambda) <= 0):
        output = Lambda
    else:
        I = torch.eye(len(theta))
        output = (I - projection_scale) @ Lambda

    return output

    
class LyAT_Controller:
    def __init__(self, config):
        self.config = config
        self.n_states = config['n_states']
        self.window_size = config['window_size']

        self.ke = config['ke']
        self.gamma = config['gamma']
        self.sigma = config['sigma']
        self.dt = config['dt']
        self.theta_bar = config['theta_bar']

        self.transformer = LyAT (
            n_states = self.n_states,            
            window_size = self.window_size,
            num_encoder_layers = config.get('num_encoder_layers'),
            num_decoder_layers = config.get('num_decoder_layers'),
            num_heads = config.get('num_heads'),
            d_ff = config.get('d_ff'),
            gamma_encoder_attn = config.get('gamma_encoder_attn'),
            beta_encoder_attn = config.get('beta_encoder_attn'),
            gamma_encoder_ff = config.get('gamma_encoder_ff'),
            beta_encoder_ff = config.get('beta_encoder_ff'),
            gamma_decoder_self = config.get('gamma_decoder_self'),
            beta_decoder_self = config.get('beta_decoder_self'),
            gamma_decoder_cross = config.get('gamma_decoder_cross'),
            beta_decoder_cross = config.get('beta_decoder_cross'),
            gamma_decoder_ff = config.get('gamma_decoder_ff'),
            beta_decoder_ff = config.get('beta_decoder_ff')
        )

        n_parameters = self.transformer.count_parameters()
        self.Gamma = torch.eye (n_parameters) * self.gamma

        self.state_history = []
        self.desired_history = []
        self.error_history = []
        self.Phi_history = []


    def update_history (self, x, xd, e, Phi):
        self.state_history.append (x.detach().clone())
        self.desired_history.append (xd.detach().clone())
        self.error_history.append (e.detach().clone())
        self.Phi_history.append (Phi.detach().clone())

        if len(self.state_history) > self.window_size:
            self.state_history.pop(0)
            self.desired_history.pop(0)
            self.error_history.pop(0)
            self.Phi_history.pop(0)

    def build_encoder_input (self):
        n_history = len(self.state_history)
        torch.manual_seed(0)
        if n_history < self.window_size:
            pad_size = self.window_size - n_history
            x_pad = [torch.randn(self.n_states) * 0.1 for _ in range(pad_size)] + self.state_history
            xd_pad = [torch.randn(self.n_states) * 0.1 for _ in range(pad_size)] + self.desired_history
            e_pad = [torch.randn(self.n_states) * 0.1 for _ in range(pad_size)] + self.error_history
        else:
            x_pad = self.state_history
            xd_pad = self.desired_history
            e_pad = self.error_history

        zeta_list = []
        for x, xd, e in zip(x_pad, xd_pad, e_pad):
            zeta_list.append(torch.cat([x, xd, e]))
        
        zeta_encoder = torch.stack(zeta_list).unsqueeze(0)

        return zeta_encoder
    
    def build_decoder_input (self):
        n_history = len(self.Phi_history)
        torch.manual_seed(0)
        if n_history < self.window_size:
            pad_size = self.window_size - n_history
            Phi_pad = [torch.randn(self.n_states) * 0.1 for _ in range(pad_size)] + self.Phi_history
        else:
            Phi_pad = self.Phi_history
        
        Phi_input = torch.stack(Phi_pad).unsqueeze(0)

        return Phi_input
    
    @torch.enable_grad()
    def compute_jacobian (self, zeta_encoder, Phi_input):
        
        for param in self.transformer.parameters():
            param.requires_grad = True
        
        n_parameters = self.transformer.count_parameters ()
        jacobian = torch.zeros(self.n_states, n_parameters)

        for i in range (self.n_states):
            self.transformer.zero_grad()
            # Fresh forward pass
            Phi = self.transformer(zeta_encoder, Phi_input).squeeze(0)
            Phi_i = Phi[i]
            
            # Backward pass
            Phi_i.backward(retain_graph=(i < self.n_states - 1))
            
            # Collect gradients from all parameters
            grads = []
            for param in self.transformer.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1).clone())
                else:
                    grads.append(torch.zeros_like(param).view(-1))
            
            if grads:
                jacobian[i, :] = torch.cat(grads)
        
        self.transformer.zero_grad()
        return jacobian

    def parameter_adaptation (self, x, t):
        xd, xd_dot = Dynamics.desired_trajectory (torch.tensor(t, dtype = torch.float32))

        e = x - xd

        zeta_encoder = self.build_encoder_input ()
        Phi_input = self.build_decoder_input ()

        # forward pass to get phi
        with torch.enable_grad ():
            Phi = self.transformer(zeta_encoder, Phi_input).squeeze(0)

        jacobian = self.compute_jacobian (zeta_encoder, Phi_input)

        theta = torch.cat([p.view(-1) for p in self.transformer.parameters()])

        projected_term = self.Gamma @ (jacobian.T @ e - self.sigma * theta)

        projected = smooth_projection (projected_term, theta, self.theta_bar)

        theta_new = theta + projected * self.dt
        # de bug
        theta_new_norm = torch.norm(theta_new).item()
        if theta_new_norm > self.theta_bar:
            print(f"\nPROJECTION FAILED at t={t:.3f}")


        idx = 0
        with torch.no_grad():
            for param in self.transformer.parameters ():
                param_size = param.numel()
                param.data = theta_new[idx:idx+param_size].view_as(param)
                idx += param_size
            
        g1 = Dynamics.control_effectiveness()
        g1_inv = torch.inverse(g1) 

        u = g1_inv @ (xd_dot - self.ke * e - Phi)
        
        self.update_history(x, xd, e, Phi)
        
        return u.detach(), Phi.detach()
    

    # ================= Simulation =================== #
class Simulation:
    def __init__ (self, config_path = 'config_LyAT.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.n_states = self.config['n_states']
        self.dt = self.config['dt']
        self.T_final = self.config['T_final']
        self.time_steps = int(self.T_final / self.dt)

        self.x = torch.zeros((self.n_states, self.time_steps))
        self.x[:, 0] = torch.tensor([0, -1, 3, -3, 3], dtype=torch.float32)

    def update_state(self, step, u, t):
        x_current = self.x[:, step - 1]
        
        f = Dynamics.drift_vector(x_current)
        g1 = Dynamics.control_effectiveness()
        g2 = Dynamics.diffusion_matrix(x_current)
        Sigma = Dynamics.covariance_matrix(torch.tensor(t, dtype=torch.float32))
        dw = torch.randn(2) * math.sqrt(torch.tensor(self.dt))
    
        self.x[:, step] = self.x[:, step - 1] + (f + g1 @ u) * self.dt + g2 @ Sigma @ dw

    def run(self): 
        controller = LyAT_Controller(self.config)
        tracking_errors = []
        control_inputs = []
        parameter_norms = []
        
        for step in range(1, self.time_steps):
            t = step * self.dt

            # Progress 
            if step % 100 == 0:
                progress = 100 * step / self.time_steps
                theta = torch.cat([p.view(-1) for p in controller.transformer.parameters()])
                param_norm = torch.norm(theta).item()
                parameter_norms.append(param_norm) 
                print(f"\rProgress: {progress:.1f}% | ||θ||: {param_norm:.2f}", end="", flush=True)

            x = self.x[:, step - 1]
            xd, xd_dot = Dynamics.desired_trajectory(torch.tensor(t, dtype=torch.float32))
            u, Phi = controller.parameter_adaptation(x, t)
            self.update_state(step, u, t)

            e = self.x[:, step] - xd
            error_norm = torch.norm(e).item()
            tracking_errors.append(error_norm)
            control_inputs.append(torch.norm(u).item())

        print(f"\n\nSimulation complete!")
        print(f"RMS tracking error: {np.sqrt(np.mean(np.square(tracking_errors))):.6f}")  
        print(f"RMS control input: {np.sqrt(np.mean(np.square(control_inputs))):.6f}") 

        self.results = {
            'x': self.x,
            'tracking_errors': tracking_errors,
            'control_inputs': control_inputs,
            'parameter_norms': parameter_norms,
            'time': np.arange(1, self.time_steps) * self.dt
        }

        return self.results
        
    def save_results(self, filename='LyAT_results.json'):
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
        plt.title('LyAT Controller: Tracking Error', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tracking_error.png', dpi=300)
        print("\nPlot saved as 'tracking_error.png'")
        plt.show()


# ================= Main =================== #
def main ():
    torch.manual_seed(42)
    np.random.seed(42)
    with open('lyapunov_adaptive_transformer/lyapunov_adaptive_transformer/config.json', 'r') as config_file: config = json.load(config_file)

    # Save config
    with open('config_LyAT.json', 'w') as f:
        json.dump(config, f, indent=4)
    print("Config file created: config_LyAT.json")
    
    # Run simulation
    sim = Simulation('config_LyAT.json')
    results = sim.run()
    
    # Save results
    sim.save_results('LyAT_results.json')

    sim.plot_results()
    
    print("\n" + "="*70)
    print("SIMULATION FINISHED!")
    print("="*70)


if __name__ == "__main__":
    main()
