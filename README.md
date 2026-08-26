# model-benchmarks
[us]- DTLN | DeepFilterNet3 | DCCRN

## Clean-input benchmark

All three models were tested with `test_voices/clean.wav` as both the input and
the clean reference. This measures how well each model preserves already-clean
speech; it is not the noisy-speech enhancement benchmark.

| Model | Backend | Processing (s) | Real-time factor | Output SNR (dB) | STOI | PESQ | Model size (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| DTLN | ONNX Runtime | 0.336 | 0.0208 | -13.06 | 0.897 | 2.004 | 6.08 |
| DeepFilterNet3 | PyTorch | 0.429 | 0.0266 | 13.83 | 0.918 | 2.032 | 8.72 |
| DCCRN | PyTorch | 1.305 | 0.0809 | -2.81 | 0.946 | 2.255 | 46.49 |

The source metrics are available in [`results/clean_input_test/`](results/clean_input_test/).

---

## Metric Definitions & Terminology

### Models & Backends
*   **DTLN (Dual-Signal Transformation LSTM Network):** A lightweight speech enhancement architecture combining Short-Time Fourier Transform (STFT) magnitude spectrograms with learnable convolutional feature representations.
*   **DeepFilterNet3:** A deep filtering framework designed for full-band (48 kHz) audio that enhances the spectral envelope and uses deep filtering to reconstruct harmonic speech structures.
*   **DCCRN (Deep Complex Convolution Recurrent Network):** A complex-domain network designed to process both the magnitude and phase information of audio simultaneously via complex convolutions and complex LSTMs.
*   **ONNX Runtime (Open Neural Network Exchange):** A high-performance inference engine optimized across hardware targets for deployed machine learning models.
*   **PyTorch:** The native Python deep learning framework used for training and experimental inference evaluation.

---

### Evaluation Columns Explained

#### 1. Processing (s)
*   **Meaning:** The wall-clock time in seconds that the model took to process the entire test audio sample.
*   **Interpretation:** Lower is faster. DTLN was the fastest at 0.336s, while DCCRN took the longest at 1.305s.

#### 2. Real-Time Factor (RTF)
*   **Formula:** $$\text{RTF} = \frac{\text{Processing Time (s)}}{\text{Total Audio Duration (s)}}$$
*   **Meaning:** A measure of processing speed relative to real-time playback.
*   **Interpretation:** An $\text{RTF} < 1.0$ means the model processes audio faster than it plays (required for real-time live transmission). An RTF of `0.0208` means the model processes a 1-second audio frame in roughly 20.8 milliseconds.

#### 3. Output SNR (Signal-to-Noise Ratio in Decibels)
*   **Meaning:** Compares the energy of the desired clean speech against any unwanted noise or distortion introduced by the model.
*   **Interpretation:** 
    *   **Positive values:** The speech signal is significantly louder and clearer than any background artifacts.
    *   **Negative values:** The model introduced noticeable distortion or mistakenly filtered out parts of the clean speech signal.

#### 4. STOI (Short-Time Objective Intelligibility)
*   **Scale:** `0.0` (completely unintelligible) to `1.0` (perfectly intelligible / 100% human speech clarity).
*   **Meaning:** Measures how easily a human listener can understand the spoken words.
*   **Interpretation:** All three models scored above `0.89`, indicating high word intelligibility preservation.

#### 5. PESQ (Perceptual Evaluation of Speech Quality)
*   **Scale:** `-0.5` to `4.5` (ITU-T P.862 standard), where `4.5` indicates crystal-clear, uncompressed speech.
*   **Meaning:** Evaluates the overall perceptual quality and naturalness of the voice, penalizing robotic or muffled artifacts.
*   **Interpretation:** Scores between `2.0` and `2.3` reflect standard telephone/radio transmission quality.

#### 6. Model Size (MB)
*   **Meaning:** The disk footprint and memory allocation required to load the model weights.
*   **Interpretation:** Critical for edge hardware deployment (e.g., Raspberry Pi, Jetson boards, DSPs). Smaller sizes mean lower RAM consumption and faster memory bus access.

---

## Results Analysis & Model Comparison

This clean-input test evaluates **transparency**: when an audio stream contains clean voice without background disturbances, the model should ideally leave the signal untouched without over-filtering.

### 1. DeepFilterNet3 (Best Overall Balance)
*   **Output SNR:** `+13.83 dB` — Successfully passed the audio with negligible distortion.
*   **Efficiency:** `8.72 MB` footprint with a fast RTF of `0.0266`.
*   **Takeaway:** Best candidate for preserving natural speech when background noise is absent.

### 2. DTLN (Fastest, High Distortion on Clean Input)
*   **Output SNR:** `-13.06 dB` — Aggressively attenuated speech harmonics, mistaking natural vocal formants for noise.
*   **Efficiency:** Smallest footprint (`6.08 MB`) and fastest runtime (`0.0208 RTF`).
*   **Takeaway:** Highly efficient, but requires suppression threshold tuning to avoid degrading clean voice inputs.

### 3. DCCRN (Highest Perceptual Quality, Computationally Heavy)
*   **Intelligibility & Quality:** Highest STOI (`0.946`) and PESQ (`2.255`).
*   **Overhead:** Largest model size (`46.49 MB`) and slowest inference time (`1.305s`, $3\times$ to $4\times$ slower than competitors).
*   **Takeaway:** High-quality processing, but poses latency and memory constraints on edge devices without quantization (e.g., INT8/FP16).

---

## Summary & Next Steps

*   **Clean Baseline Winner:** **DeepFilterNet3** provides the highest fidelity and transparency on clean voice signals while remaining lightweight.
*   **Next Milestone:** Run the noisy benchmark suite across the mixed construction audio files (`dirty_construction_-5dB.wav`, `dirty_construction_0dB.wav`, `dirty_construction_5dB.wav`, `dirty_construction_10dB.wav`) to evaluate noise suppression efficacy and dynamic noise rejection.
