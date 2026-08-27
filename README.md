# model-benchmarks

DTLN | DeepFilterNet3 | DCCRN | GTCRN

## Clean-input benchmark

The models were tested with `test_voices/clean.wav` as both the input and the clean reference. This measures how well each model preserves already-clean speech; it is not the noisy-speech enhancement benchmark.

| Model          | Backend      | Processing (s) | Real-time factor | Output SNR (dB) |  STOI |  PESQ | Model size (MB) |
| -------------- | ------------ | -------------: | ---------------: | --------------: | ----: | ----: | --------------: |
| DTLN           | ONNX Runtime |          0.336 |           0.0208 |          -13.06 | 0.897 | 2.004 |            6.08 |
| DeepFilterNet3 | PyTorch      |          0.429 |           0.0266 |           13.83 | 0.918 | 2.032 |            8.72 |
| DCCRN          | PyTorch      |          1.305 |           0.0809 |           -2.81 | 0.946 | 2.255 |           46.49 |

The source metrics are available in [`results/clean_input_test/`](results/clean_input_test/).

---

## Noisy-input benchmark — Construction noise

The noisy-input benchmark evaluates how effectively each model enhances speech when the input contains construction noise at different signal-to-noise ratios (SNRs). The clean reference is `test_voices/clean.wav`.

The benchmark was run on four noisy inputs:

* `dirty_construction_-5dB.wav`
* `dirty_construction_0dB.wav`
* `dirty_construction_5dB.wav`
* `dirty_construction_10dB.wav`

### Construction noise results

| Noise level | Model          | Output SI-SDR (dB) | SI-SDR improvement (dB) | Output SNR (dB) | SNR improvement (dB) |      STOI |      PESQ | Processing (s) |       RTF | Model size (MB) |
| ----------- | -------------- | -----------------: | ----------------------: | --------------: | -------------------: | --------: | --------: | -------------: | --------: | --------------: |
| -5 dB       | DeepFilterNet3 |               8.98 |                   13.98 |            8.89 |                13.89 |     0.646 |     1.245 |           1.98 |     0.123 |            8.72 |
| -5 dB       | GTCRN          |          **11.80** |               **16.79** |       **11.76** |            **16.76** |     0.630 | **1.455** |       **0.91** | **0.056** |        **0.58** |
| 0 dB        | DeepFilterNet3 |              10.12 |                   10.12 |            9.95 |                 9.95 | **0.674** |     1.328 |           5.68 |     0.352 |            8.72 |
| 0 dB        | GTCRN          |          **13.29** |               **13.28** |       **13.18** |            **13.18** |     0.665 | **1.570** |       **1.08** | **0.067** |        **0.58** |
| +5 dB       | DeepFilterNet3 |              11.09 |                    6.09 |           10.95 |                 5.95 | **0.708** |     1.494 |          22.53 |     1.397 |            8.72 |
| +5 dB       | GTCRN          |          **14.41** |                **9.41** |       **14.27** |             **9.27** |     0.695 | **1.650** |       **0.75** | **0.047** |        **0.58** |
| +10 dB      | DeepFilterNet3 |              11.70 |                    1.70 |           11.64 |                 1.64 | **0.742** |     1.608 |          15.93 |     0.988 |            8.72 |
| +10 dB      | GTCRN          |          **15.07** |                **5.07** |       **14.89** |             **4.89** |     0.738 | **1.699** |       **0.88** | **0.055** |        **0.58** |

### Analysis

#### 1. GTCRN — Strong noisy-speech enhancement

GTCRN consistently achieved higher SI-SDR and SNR than DeepFilterNet3 at all four construction-noise levels.

* **SI-SDR:** GTCRN achieved a maximum improvement of **16.79 dB** at -5 dB input SNR, compared with **13.98 dB** for DeepFilterNet3.
* **SNR:** GTCRN achieved a maximum improvement of **16.76 dB** at -5 dB input SNR, compared with **13.89 dB** for DeepFilterNet3.
* **PESQ:** GTCRN achieved higher PESQ at every tested noise level.
* **Efficiency:** GTCRN uses a much smaller model footprint of approximately **0.58 MB** and maintained an RTF below **0.07** for all four tests.

GTCRN therefore provides the strongest combination of noise suppression, perceptual quality, model size, and inference efficiency in this construction-noise benchmark.

#### 2. DeepFilterNet3 — Strong intelligibility preservation

DeepFilterNet3 produced slightly higher STOI scores than GTCRN at all tested noise levels, indicating slightly better preservation of speech intelligibility according to this metric.

However, it required a substantially larger model footprint of **8.72 MB**. Its processing time was also considerably higher in the +5 dB and +10 dB tests.

#### 3. Overall comparison

The noisy-input benchmark shows that **GTCRN is the stronger model for construction-noise suppression** in this evaluation.

GTCRN provides higher SI-SDR, SNR, and PESQ while using approximately **15× less model storage** than DeepFilterNet3 and achieving substantially lower real-time factors.

DeepFilterNet3 remains competitive in STOI and provides good speech intelligibility preservation, but GTCRN offers a stronger combination of enhancement performance and computational efficiency for this test.

### Source metrics

The detailed benchmark outputs are available in:

* [`results/gtcrn_-5dB/`](results/gtcrn_-5dB/)
* [`results/gtcrn_0dB/`](results/gtcrn_0dB/)
* [`results/gtcrn_5dB/`](results/gtcrn_5dB/)
* [`results/gtcrn_10dB/`](results/gtcrn_10dB/)

Each directory contains the enhanced WAV files and the corresponding `metrics.csv` and `metrics.json` benchmark reports.

---

## Metric Definitions & Terminology

### Models & Backends

* **DTLN (Dual-Signal Transformation LSTM Network):** A lightweight speech enhancement architecture combining Short-Time Fourier Transform (STFT) magnitude spectrograms with learnable convolutional feature representations.

* **DeepFilterNet3:** A deep filtering framework designed for full-band (48 kHz) audio that enhances the spectral envelope and uses deep filtering to reconstruct harmonic speech structures.

* **DCCRN (Deep Complex Convolution Recurrent Network):** A complex-domain network designed to process both magnitude and phase information of audio simultaneously via complex convolutions and complex LSTMs.

* **GTCRN:** An ultra-lightweight speech enhancement network using ShuffleNetV2-based feature extraction, Subband Feature Extraction (SFE), Temporal Recurrent Attention (TRA), and dual-path gated recurrent neural network components. The evaluated DNS3 checkpoint is approximately 0.58 MB.

* **ONNX Runtime:** A high-performance inference engine optimized across hardware targets for deployed machine learning models.

* **PyTorch:** The native Python deep learning framework used for training and experimental inference evaluation.

---

### Evaluation Columns Explained

#### 1. Processing (s)

* **Meaning:** The wall-clock time in seconds that the model took to process the entire test audio sample.
* **Interpretation:** Lower is faster.

#### 2. Real-Time Factor (RTF)

**Formula:**

$$
\text{RTF} = \frac{\text{Processing Time (s)}}{\text{Total Audio Duration (s)}}
$$

An RTF below 1.0 means the model processes audio faster than real-time playback.

#### 3. Output SNR (Signal-to-Noise Ratio)

Output SNR compares the energy of the desired clean speech against the error between the clean reference and enhanced signal.

Higher values indicate better agreement with the clean reference under this evaluation.

#### 4. SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)

SI-SDR measures the quality of the enhanced signal relative to the clean reference while being invariant to overall signal scaling.

Higher SI-SDR and SI-SDR improvement indicate better enhancement performance.

#### 5. STOI (Short-Time Objective Intelligibility)

STOI is an objective measure of speech intelligibility. Values closer to 1 indicate better intelligibility preservation.

#### 6. PESQ (Perceptual Evaluation of Speech Quality)

PESQ is an objective perceptual speech-quality metric. Higher values generally indicate better perceived speech quality.

#### 7. Model Size (MB)

Model size represents the disk footprint of the model weights/checkpoint used for inference.

Smaller models are advantageous for memory-constrained and edge deployments.

---

## Results Analysis & Model Comparison

The clean-input test evaluates **transparency**: when an audio stream contains clean voice without background disturbances, the model should ideally leave the signal relatively unchanged without excessive filtering.

### 1. DeepFilterNet3 — Strong clean-input balance

DeepFilterNet3 achieved an output SNR of **13.83 dB**, STOI of **0.918**, and PESQ of **2.032** on the clean-input test while maintaining an RTF of **0.0266**.

### 2. DTLN — Fastest clean-input inference

DTLN had the lowest processing time and RTF in the clean-input benchmark, but its output SNR was negative, indicating substantial distortion under this particular test configuration.

### 3. DCCRN — Strong perceptual metrics

DCCRN achieved the highest STOI (**0.946**) and PESQ (**2.255**) in the clean-input benchmark, but it also had the largest model footprint (**46.49 MB**) and highest processing time (**1.305 s**).

### 4. GTCRN — Lightweight noisy-input specialist

GTCRN demonstrated particularly strong performance in the construction-noise benchmark. It achieved the highest SI-SDR and SNR improvements among the compared noisy-input models while maintaining a model size of only **0.58 MB** and RTF below **0.07** across all tested SNR levels.

---

## Summary & Next Steps

### Current findings

* **Clean-input benchmark:** DeepFilterNet3 provides a strong balance of clean-speech preservation, perceptual quality, and efficiency.
* **Construction-noise benchmark:** GTCRN provides the strongest SI-SDR, SNR, and PESQ performance among the currently evaluated noisy-input models.
* **Efficiency:** GTCRN is substantially smaller than DeepFilterNet3, at approximately **0.58 MB versus 8.72 MB**.
* **Real-time processing:** GTCRN maintained an RTF below **0.07** across all four construction-noise conditions.

### Next steps

* Add DTLN and DCCRN to the same construction-noise benchmark if compatible checkpoints/configurations are available.
* Evaluate additional noise types and SNR conditions.
* Aggregate results across multiple speech samples rather than a single clean reference.
* Evaluate the models on edge hardware to measure practical latency and memory usage.
