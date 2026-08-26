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
