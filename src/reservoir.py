import numpy as np
from brian2 import *

# suppress Brian2's verbose output
prefs.codegen.target = 'numpy'

def build_reservoir(N=1000, p_conn=0.1, spectral_radius=0.9,
                    tau_m=20, ei_ratio=0.8, seed=42):
    """
    Build a random recurrent LIF reservoir.
    
    N              : number of neurons
    p_conn         : connection probability (sparsity)
    spectral_radius: scales recurrent weights
    tau_m          : membrane time constant (ms)
    ei_ratio       : fraction of excitatory neurons
    """
    np.random.seed(seed)
    n_exc = int(N * ei_ratio)
    n_inh = N - n_exc

    # --- Build recurrent weight matrix W (N x N) ---
    W = np.random.randn(N, N)
    # Apply sparsity: zero out connections with prob (1 - p_conn)
    mask = np.random.rand(N, N) < p_conn
    np.fill_diagonal(mask, False)      # no self-connections
    W *= mask.astype(float)
    # Inhibitory neurons contribute negative weights
    W[:, n_exc:] *= -1                 # columns n_exc: are inhibitory
    # Scale to desired spectral radius
    eig_max = np.max(np.abs(np.linalg.eigvals(W)))
    if eig_max > 0:
        W *= (spectral_radius / eig_max)

    return W, n_exc, n_inh


def run_reservoir(spike_input, W, tau_m=20.0, dt_ms=1.0,
                  v_th = 0.5, v_reset=0.0, tau_ref=2.0):
    """
    Simulate the reservoir given a spike input.
    
    spike_input : (T, n_input) binary array
    W           : (N, N) recurrent weight matrix
    Returns     : state vector of shape (N,) — mean firing rates
    """
    T, n_in = spike_input.shape
    N = W.shape[0]
    dt = dt_ms * 1e-3          # convert ms to seconds

    # Input weights: random, shape (N, n_input)
    W_in = np.random.randn(N, n_in) * 5.0

    V         = np.zeros(N)    # membrane potentials
    spk_count = np.zeros(N)    # spike count per neuron
    refractory = np.zeros(N)   # remaining refractory time

    alpha = np.exp(-dt / (tau_m * 1e-3))   # decay factor

    for t in range(T):
        # Input current from spike_input at this timestep
        I_in  = W_in @ spike_input[t].astype(float)
        # Recurrent current from spikes last timestep
        I_rec = W @ (V >= v_th).astype(float)

        # Update membrane potential (LIF dynamics)
        V = alpha * V + (1 - alpha) * (I_in + I_rec)

        # Refractory: force V=0 for neurons still in refractory period
        V[refractory > 0] = v_reset
        refractory = np.maximum(refractory - dt_ms, 0)

        # Fire & reset
        fired = V >= v_th
        spk_count += fired.astype(float)
        V[fired]          = v_reset
        refractory[fired] = tau_ref

    # State vector = mean firing rate (spikes per timestep)
    state = spk_count / T
    return state


def encode_dataset_to_states(X_spikes, W, n_jobs=1, **kwargs):
    """
    Run every spike-encoded signal through the reservoir.
    X_spikes: list or array of (T, 2) spike arrays
    Returns:  (n_samples, N) state matrix
    """
    states = []
    for i, spk in enumerate(X_spikes):
        s = run_reservoir(spk, W, **kwargs)
        states.append(s)
        if i % 1000 == 0:
            print(f"  {i}/{len(X_spikes)} done")
    return np.array(states)