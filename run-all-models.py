#!/usr/bin/env python3
"""Run the three available speech-enhancement models and benchmark them."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def mono(audio):
    """Convert stereo audio to mono audio."""
    return audio.mean(axis=1).astype("float32") if audio.ndim == 2 else audio.astype("float32")


def resample(audio, source_rate, target_rate):
    """Convert audio to the sample rate required by a model or metric."""
    if source_rate == target_rate:
        return audio.astype("float32")
    return resample_poly(audio, target_rate, source_rate).astype("float32")


def rms(audio):
    """Calculate average audio energy."""
    return np.sqrt(np.mean(audio ** 2) + 1e-12)


def snr_db(clean, estimate):
    """Calculate SNR using clean speech as the reference."""
    error = clean - estimate
    return float(10 * np.log10((np.sum(clean ** 2) + 1e-8) /
                               (np.sum(error ** 2) + 1e-8)))


def si_sdr(clean, estimate):
    """Calculate scale-invariant SDR."""
    clean = clean - clean.mean()
    estimate = estimate - estimate.mean()
    scale = np.dot(estimate, clean) / (np.dot(clean, clean) + 1e-8)
    target = scale * clean
    error = estimate - target
    return float(10 * np.log10((np.sum(target ** 2) + 1e-8) /
                               (np.sum(error ** 2) + 1e-8)))


def quality_metrics(clean, noisy, enhanced):
    """Calculate objective speech-quality metrics at 16 kHz."""
    from pesq import pesq
    from pystoi import stoi

    length = min(len(clean), len(noisy), len(enhanced))
    clean, noisy, enhanced = clean[:length], noisy[:length], enhanced[:length]

    input_snr = snr_db(clean, noisy)
    output_snr = snr_db(clean, enhanced)
    input_sisdr = si_sdr(clean, noisy)
    output_sisdr = si_sdr(clean, enhanced)

    return {
        "input_snr_db": input_snr,
        "output_snr_db": output_snr,
        "snr_improvement_db": output_snr - input_snr,
        "input_si_sdr_db": input_sisdr,
        "output_si_sdr_db": output_sisdr,
        "si_sdr_improvement_db": output_sisdr - input_sisdr,
        "stoi": float(stoi(clean, enhanced, 16000, extended=False)),
        "pesq": float(pesq(16000, clean, enhanced, "wb")),
    }


def size_bytes(path):
    """Return the size of a model file or model directory."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run_dtln(audio, model_dir):
    """Run DTLN through its existing ONNX Runtime streaming wrapper."""
    root = Path(__file__).parent / "model-files/DTLN_Real_Time_Speech-Enhancement"
    sys.path.insert(0, str(root))
    from src.onnx_infer import StreamingEnhancer

    enhancer = StreamingEnhancer(str(model_dir))
    return enhancer.enhance(audio), enhancer.sample_rate, size_bytes(model_dir), "onnxruntime"


def run_deepfilternet(audio, input_rate, model_dir):
    """Run DeepFilterNet3 using its current Python inference API."""
    root = Path(__file__).parent / "model-files/DeepFilterNet/DeepFilterNet"
    sys.path.insert(0, str(root))
    import torch
    from df.enhance import enhance, init_df

    model, state, _, _ = init_df(model_base_dir=str(model_dir), log_file=None)
    target_rate = state.sr()
    tensor = torch.from_numpy(resample(audio, input_rate, target_rate)).unsqueeze(0)

    with torch.no_grad():
        output = enhance(model, state, tensor, pad=True).squeeze(0).numpy()

    return output, target_rate, size_bytes(model_dir), "pytorch"


def run_dccrn(audio, input_rate, checkpoint):
    """Run DCCRN using a compatible PyTorch checkpoint."""
    root = Path(__file__).parent / "model-files/DCCRN"
    sys.path.insert(0, str(root))
    import torch
    from model import DCCRN

    audio = resample(audio, input_rate, 16000)
    model = DCCRN()
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        output = model(torch.from_numpy(audio).unsqueeze(0)).squeeze(0).numpy()

    return output, 16000, size_bytes(checkpoint), "pytorch"


def main():
    parser = argparse.ArgumentParser(
        description="Run DTLN, DeepFilterNet3 and DCCRN on one noisy WAV."
    )
    parser.add_argument("--input", default="test_voices/dirty_construction_0dB.wav")
    parser.add_argument("--clean", default="test_voices/clean.wav")
    parser.add_argument("--out-dir", default="results/all_models")
    parser.add_argument(
        "--dtln-dir",
        default="model-files/DTLN_Real_Time_Speech-Enhancement/onnx"
    )
    parser.add_argument("--deepfilternet-dir", required=True)
    parser.add_argument("--dccrn-checkpoint", required=True)
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    noisy, input_rate = sf.read(args.input, dtype="float32")
    noisy = mono(noisy)
    noisy_16k = resample(noisy, input_rate, 16000)

    clean, clean_rate = sf.read(args.clean, dtype="float32")
    clean = resample(mono(clean), clean_rate, 16000)

    jobs = [
        ("DTLN", lambda: run_dtln(noisy_16k, args.dtln_dir), args.dtln_dir),
        ("DeepFilterNet3", lambda: run_deepfilternet(
            noisy, input_rate, args.deepfilternet_dir), args.deepfilternet_dir),
        ("DCCRN", lambda: run_dccrn(
            noisy, input_rate, args.dccrn_checkpoint), args.dccrn_checkpoint),
    ]

    results = []

    for name, runner, model_path in jobs:
        started = time.perf_counter()
        enhanced, model_rate, model_bytes, backend = runner()
        elapsed = time.perf_counter() - started
        enhanced_16k = resample(enhanced, model_rate, 16000)

        output_path = output_dir / f"{name}-filtered.wav"
        sf.write(output_path, enhanced_16k, 16000)

        row = {
            "model": name,
            "backend": backend,
            "input_file": args.input,
            "clean_reference": args.clean,
            "duration_seconds": len(noisy) / input_rate,
            "input_sample_rate": input_rate,
            "model_sample_rate": model_rate,
            "processing_seconds": elapsed,
            "real_time_factor": elapsed / (len(noisy) / input_rate),
            "model_size_bytes": model_bytes,
            "model_path": str(model_path),
            "output_file": str(output_path),
        }

        try:
            row.update(quality_metrics(clean, noisy_16k, enhanced_16k))
        except Exception as error:
            row["metric_error"] = str(error)

        results.append(row)
        print(f"{name}: wrote {output_path}")

    fields = sorted({field for row in results for field in row})
    with open(output_dir / "metrics.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"Reports written to {output_dir}")


if __name__ == "__main__":
    main()
