import numpy as np

def normalise(x):
    """Normalise signal to [0, 1]."""
    x = x - x.min()
    r = x.max()
    return x / r if r > 0 else x

def rate_encode(iq_signal, lam_max=0.9):
    """
    Rate coding: amplitude → Poisson firing rate.
    Input:  (T, 2) float IQ signal
    Output: (T, 2) binary spike array
    """
    T = iq_signal.shape[0]
    spikes = np.zeros_like(iq_signal, dtype=np.int8)
    for ch in range(2):
        rates = normalise(np.abs(iq_signal[:, ch])) * lam_max
        spikes[:, ch] = (np.random.rand(T) < rates).astype(np.int8)
    return spikes

def ttfs_encode(iq_signal, T_max=None):
    """
    Time-to-first-spike: amplitude → spike delay.
    Large amplitude → spike fires early.
    Output: (T, 2) binary array with at most 1 spike per channel.
    """
    T = iq_signal.shape[0]
    if T_max is None:
        T_max = T
    spikes = np.zeros((T, 2), dtype=np.int8)
    for ch in range(2):
        norm = normalise(np.abs(iq_signal[:, ch]))
        # for each sample, compute its firing time
        # here we encode each sample independently as 1 neuron
        # simplification: take the mean amplitude per 8-sample block
        block = 8
        for b in range(T // block):
            amp = norm[b*block:(b+1)*block].mean()
            t_fire = int((1.0 - amp) * (block - 1))
            idx = b * block + t_fire
            if idx < T:
                spikes[idx, ch] = 1
    return spikes

def hybrid_encode(iq_signal, theta=0.5, lam_max=0.9):
    """
    Hybrid: high-amplitude → TTFS (efficient),
            low-amplitude  → rate coding (robust).
    """
    T = iq_signal.shape[0]
    spikes = np.zeros((T, 2), dtype=np.int8)
    for ch in range(2):
        norm = normalise(np.abs(iq_signal[:, ch]))
        high = norm >= theta
        low  = ~high
        # rate coding for low-amplitude parts
        rates = norm * lam_max
        rate_spikes = (np.random.rand(T) < rates).astype(np.int8)
        spikes[:, ch][low]  = rate_spikes[low]
        # TTFS for high-amplitude parts
        block = 8
        for b in range(T // block):
            start, end = b*block, (b+1)*block
            if norm[start:end].mean() >= theta:
                amp    = norm[start:end].mean()
                t_fire = int((1.0 - amp) * (block-1))
                idx    = start + t_fire
                if idx < T:
                    spikes[idx, ch] = 1
    return spikes