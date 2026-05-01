import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import convolve
import sympy as sp

print("=" * 50)
print(" DSP Project - Audio Signal Processing")
print("=" * 50)

audio_file = "vRcbcpKaTS0.wav"

fs, data = wavfile.read(audio_file)

if len(data.shape) > 1:
    data = data[:, 0]

data = data.astype(float)
# Normalize
data = data / np.max(np.abs(data))
duration = len(data) / fs

n = np.arange(len(data))
time = n / fs


# ========================================
# Elementary Discrete-Time Signals
# ========================================
def elementary_signals():

    n_small = np.arange(-10, 100)

    unit_sample = np.where(n_small == 0, 1, 0)
    unit_step = np.where(n_small >= 0, 1, 0)
    unit_ramp = np.where(n_small >= 0, n_small, 0)

    a = 0.9
    exponential = a**n_small

    return n_small, unit_sample, unit_step, unit_ramp, exponential


# ========================================
# Sampling & Quantization
# ========================================
def sampling_quantization(signal, bits=4):

    levels = 2**bits
    max_val = np.max(signal)
    min_val = np.min(signal)

    delta = (max_val - min_val) / levels
    quantized = np.round((signal - min_val) / delta) * delta + min_val

    print(f"   Bits: {bits}, Levels: {levels}, Delta: {delta:.6f}")
    return quantized, delta


# ========================================
# Convolution
# ========================================
def convolution(input_signal, h):

    signal_l = len(input_signal)
    h_l = len(h)
    y_l = signal_l + h_l - 1

    y = np.zeros(y_l)

    for n in range(y_l):
        for k in range(signal_l):
            if (n - k) >= 0 and (n - k) < h_l:
                y[n] += input_signal[k] * h[n - k]

    return y


def add_echo_effect(signal, delay_samples=2000, decay=0.6):

    # impulse response  echo
    h = np.zeros(delay_samples + 1)
    h[0] = 1.0
    h[delay_samples] = decay

    output = convolve(signal, h, mode="same")

    # Normalize
    output = output / np.max(np.abs(output))
    return output


# ========================================
# Z-Transform
# ========================================
def z_transform(signal, num_samples=10):
    """Z-Transform"""

    z = sp.symbols("z")

    x = signal[:num_samples]
    index = np.arange(num_samples)

    X_z = 0
    for i in range(len(x)):
        X_z += x[i] * z ** (-index[i])

    print("X(z) =", X_z)
    return X_z


# ========================================
# Fourier Transform (FFT)
# ========================================
def fourier(signal, sample_rate):
    """FFT"""
    N = len(signal)

    yf = fft(signal)
    xf = fftfreq(N, 1 / sample_rate)

    xf = xf[: N // 2]
    yf = 2.0 / N * np.abs(yf[: N // 2])

    return xf, yf


# ========================================
# Signal Operations
# ========================================
def signal_operations(signal):

    # Time Shift
    shift = 1000
    shifted = np.zeros_like(signal)
    if shift > 0 and shift < len(signal):
        shifted[shift:] = signal[:-shift]

    # Amplitude Scaling
    scaled = signal * 0.5

    # Accumulation
    accumulated = np.cumsum(signal)
    accumulated = accumulated / np.max(np.abs(accumulated))

    return shifted, scaled, accumulated


n_small, unit_sample, unit_step, unit_ramp, exponential = elementary_signals()

quantized, delta = sampling_quantization(data, bits=4)

echo_signal = add_echo_effect(data, delay_samples=int(0.15 * fs), decay=0.5)

z_transform_result = z_transform(data, num_samples=8)

xf, yf = fourier(data, fs)

shifted, scaled, accumulated = signal_operations(data)
# ========================================
# Matplotlib
# ========================================
fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "DSP Project - Audio Signal Processing",
    fontsize=16,
    fontweight="bold",
)

# Row 1: Original + Elementary
plt.subplot(4, 3, 1)
plt.plot(time[:3000], data[:3000], "b", linewidth=0.7)
plt.title("1. Original Audio Signal", fontweight="bold")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

plt.subplot(4, 3, 2)
plt.stem(n_small[:40], unit_sample[:40])
plt.title("Unit Sample", fontweight="bold")
plt.xlabel("n")
plt.grid(True, alpha=0.3)

plt.subplot(4, 3, 3)
plt.stem(n_small[:40], unit_step[:40])
plt.title(" Unit Step ", fontweight="bold")
plt.xlabel("n")
plt.grid(True, alpha=0.3)

# Row 2: Quantization + Unit Ramp + Exponential
plt.subplot(4, 3, 4)
plt.plot(n[:500], data[:500], "b-", label="Original", linewidth=1, alpha=0.7)
plt.plot(n[:500], quantized[:500], "r-", label="Quantized", linewidth=1.5)
plt.title(f" Quantization ({4} bits)", fontweight="bold")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(4, 3, 5)
plt.stem(n_small[:25], unit_ramp[:25])
plt.title(" Unit Ramp r[n]", fontweight="bold")
plt.xlabel("n")
plt.grid(True, alpha=0.3)

plt.subplot(4, 3, 6)
plt.stem(n_small[:25], exponential[:25])
plt.title(" Exponential (0.9)^n", fontweight="bold")
plt.xlabel("n")
plt.grid(True, alpha=0.3)

# Row 3: Convolution + FFT
plt.subplot(4, 3, 7)
plt.plot(time[:3000], echo_signal[:3000], "g", linewidth=0.7)
plt.title(" Echo Effect (Convolution)", fontweight="bold")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

plt.subplot(4, 3, 8)
plt.plot(xf[: len(xf) // 5], yf[: len(yf) // 5])
plt.title(" Frequency Spectrum (FFT)", fontweight="bold")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True, alpha=0.3)

plt.subplot(4, 3, 9)
error = data - quantized
plt.plot(n[:1000], error[:1000], "orange", linewidth=0.6)
plt.title("Quantization Error", fontweight="bold")
plt.xlabel("Sample")
plt.ylabel("Error")
plt.grid(True, alpha=0.3)

# Row 4: Signal Operations
plt.subplot(4, 3, 10)
plt.plot(time[:3000], shifted[:3000], "purple", linewidth=0.7)
plt.title(" Time Shift", fontweight="bold")
plt.xlabel("Time (s)")
plt.grid(True, alpha=0.3)

plt.subplot(4, 3, 11)
plt.plot(time[:3000], scaled[:3000], "brown", linewidth=0.7)
plt.title(" Amplitude Scale (×0.5)", fontweight="bold")
plt.xlabel("Time (s)")
plt.grid(True, alpha=0.3)

plt.subplot(4, 3, 12)
plt.plot(time[:3000], accumulated[:3000], "red", linewidth=0.7)
plt.title(" Accumulation", fontweight="bold")
plt.xlabel("Time (s)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# save opertion

# plt.savefig("dsp_results.png", dpi=200, bbox_inches="tight")

# save_choice = input("save Echo and Quantized audio(y/n): ")

# if save_choice.lower() == "y":
#     # save Quantized
#     quantized_int = np.clip(quantized * 32767, -32768, 32767).astype(np.int16)
#     wavfile.write("output_quantized.wav", fs, quantized_int)
#     print(" output_quantized.wav")

#     # save Echo
#     echo_int = np.clip(echo_signal * 32767, -32768, 32767).astype(np.int16)
#     wavfile.write("output_echo.wav", fs, echo_int)
#     print(" output_echo.wav")
