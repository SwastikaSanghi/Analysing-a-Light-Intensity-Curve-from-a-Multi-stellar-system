import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows

# -------------------------
# 0. Parameters
# -------------------------
input_file = "EPIC249432662_lightcurve Time-Flux BE.csv"  # contains 'time' and 'normalized_flux'
min_peak_prominence = 0.01   # sensitivity for peak detection
pad_factor = 4               # zero-padding factor for frequency resolution

# -------------------------
# 1. Load Data
# -------------------------
df = pd.read_csv(input_file)
time = np.array(df['TIME'])
flux = np.array(df['PDCSAP_FLUX'])

# Basic cleaning
mask = np.isfinite(time) & np.isfinite(flux)
time = time[mask]
flux = flux[mask]

# -------------------------
# 2. Compute sampling interval (assume uniform sampling)
# -------------------------
dt = time[1] - time[0]
if not np.allclose(np.diff(time), dt, rtol=1e-3):
    print("⚠️ Warning: Data may not be perfectly uniform. FFT assumes constant dt.")

# Detrend: remove mean
flux = flux - np.mean(flux)

# Apply Hann window
window = windows.hann(len(flux))
flux_windowed = flux * window

# -------------------------
# 3. FFT and Power Spectrum
# -------------------------
N = len(flux)
Npad = int(2**np.ceil(np.log2(N * pad_factor)))  # next power of two for speed

fft_vals = np.fft.rfft(flux_windowed, n=Npad)
freqs = np.fft.rfftfreq(Npad, dt)

power = np.abs(fft_vals)**2
power /= np.max(power)  # normalize

# -------------------------
# 4. Find Peaks
# -------------------------
freq_mask = freqs > 0
freqs = freqs[freq_mask]
power = power[freq_mask]

peaks, props = find_peaks(power, prominence=min_peak_prominence)
peak_freqs = freqs[peaks]
peak_powers = power[peaks]
peak_periods = 1.0 / peak_freqs

# Sort by descending power
order = np.argsort(-peak_powers)
peak_freqs = peak_freqs[order]
peak_powers = peak_powers[order]
peak_periods = peak_periods[order]

# -------------------------
# 5. Plot Spectrum
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(freqs, power, lw=1)
plt.scatter(peak_freqs, peak_powers, color='C1', zorder=3, label='Detected peaks')
plt.xlabel("Frequency (1 / time unit)")
plt.ylabel("Normalized Power")
plt.title("Fourier Power Spectrum (Direct FFT)")
plt.legend()

# Annotate top 5 peaks
for i, (f, p) in enumerate(zip(peak_freqs[:5], peak_powers[:5])):
    per = 1.0 / f
    plt.annotate(f"P{i+1} = {per:.4f}", xy=(f, p),
                 xytext=(5, -10), textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.show()

# -------------------------
# 6. Print Results
# -------------------------
print("\nTop detected periods (time units):")
for i, (p, f, powr) in enumerate(zip(peak_periods[:10], peak_freqs[:10], peak_powers[:10])):
    print(f"{i+1:2d}: Period = {p:.6f}  (Freq = {f:.6f}, Power = {powr:.4f})")

if len(peak_periods) > 1:
    print("\nNote: check if peaks correspond to harmonics or aliases.")