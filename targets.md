# DRDO ANC — Benchmark & Requirements Checklist

## 1. Explicit Numerical Targets

These are the three numerical targets explicitly stated in the problem statement:

| Metric | Target | Purpose |
|---|---:|---|
| SNR | **> 15 dB** | Noise suppression / signal quality |
| STOI | **> 0.85** | Speech intelligibility |
| PESQ | **> 2.5** | Perceived speech quality |

> These are targets, not results. Do not claim they are achieved until they are actually measured.

---

## 2. Required System Capabilities

The system must demonstrate effective handling of:

### Stationary Noise

Examples:

- Vehicle engines
- Machinery
- HVAC
- Continuous background noise

### Non-Stationary Noise

Examples:

- Helicopter rotor
- Drone
- Changing engine noise
- Sirens
- Wind

### Impulsive Noise

Examples:

- Gunshots
- Artillery
- Explosions
- Other sudden acoustic events

---

## 3. Real-Time Requirements

The problem statement requires:

- Real-time inference
- Low latency suitable for communication
- Embedded/edge deployment
- Live microphone/headset integration

The problem statement does **not** specify exact numerical requirements for:

- Maximum latency
- CPU utilization
- RAM usage
- Power consumption
- Real-time factor

Therefore, these should be measured and reported rather than treated as official requirements.

Recommended measurements:

```text
End-to-end latency
Model inference latency
Preprocessing latency
Postprocessing latency
Real-Time Factor (RTF)
Average CPU utilization
Peak CPU utilization
RAM usage
Power consumption