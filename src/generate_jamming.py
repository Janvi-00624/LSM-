import numpy as np

N = 1024        # samples per signal
fs = 1.0        # normalised sample rate
t  = np.arange(N) / fs

def add_awgn(signal, snr_db):
    """Add noise so signal has a given SNR in dB."""
    sig_power = np.mean(np.abs(signal)**2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(N) + 1j * np.random.randn(N))
    return signal + noise

def tone_jamming(snr_db):
    f_j = np.random.uniform(0.1, 0.4)   # random jam frequency
    s = np.exp(1j * 2 * np.pi * f_j * t)
    return add_awgn(s, snr_db)

def sweep_jamming(snr_db):
    f0 = np.random.uniform(0.05, 0.2)
    k  = np.random.uniform(0.001, 0.005)
    s  = np.exp(1j * 2 * np.pi * (f0*t + 0.5*k*t**2))
    return add_awgn(s, snr_db)

def noise_jamming(snr_db):
    s = np.random.randn(N) + 1j * np.random.randn(N)
    return add_awgn(s, snr_db)

def repeat_jamming(snr_db):
    # simulate a captured signal replayed with delay
    delay = np.random.randint(10, 100)
    f_c   = np.random.uniform(0.1, 0.3)
    orig  = np.exp(1j * 2 * np.pi * f_c * t)
    s     = np.zeros(N, dtype=complex)
    s[delay:] = orig[:N-delay]
    return add_awgn(s, snr_db)

def pulse_jamming(snr_db):
    pulse_w = np.random.randint(20, 80)
    period  = np.random.randint(100, 200)
    s = np.zeros(N, dtype=complex)
    for start in range(0, N, period):
        end = min(start + pulse_w, N)
        s[start:end] = 1.0
    return add_awgn(s, snr_db)

def barrage_jamming(snr_db):
    # wideband noise — just high-power noise across whole band
    s = (np.random.randn(N) + 1j*np.random.randn(N)) * 2.0
    return add_awgn(s, snr_db)

JAM_FUNCS  = [tone_jamming, sweep_jamming, noise_jamming,
              repeat_jamming, pulse_jamming, barrage_jamming]
JAM_LABELS = ['tone','sweep','noise','repeat','pulse','barrage']

def build_dataset(n_per_class=8000, snr_levels=None):
    if snr_levels is None:
        snr_levels = range(-10, 25, 5)
    X, Y, Z = [], [], []
    for snr in snr_levels:
        for label_idx, fn in enumerate(JAM_FUNCS):
            for _ in range(n_per_class // len(list(snr_levels))):
                sig = fn(snr)
                # stack I and Q as 2-channel: shape (1024, 2)
                iq  = np.stack([sig.real, sig.imag], axis=-1)
                X.append(iq)
                Y.append(label_idx)
                Z.append(snr)
    return np.array(X), np.array(Y), np.array(Z)

if __name__ == '__main__':
    X, Y, Z = build_dataset()
    np.save('data/jam_X.npy', X)
    np.save('data/jam_Y.npy', Y)
    np.save('data/jam_Z.npy', Z)
    print(f"Dataset saved: {X.shape}")