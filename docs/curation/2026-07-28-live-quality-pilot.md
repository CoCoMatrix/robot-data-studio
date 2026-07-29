# Bounded live-quality pilot and fast-path benchmark

Only official V3 Human episodes `1006561`, `1006564`, `1006570` and Robot
episodes `0`, `1`, `2` were used. Held-out paths, full scans, training, DLC and
source-media mutation remained prohibited.

## Real evidence

- C1: DashScope `qwen3.7-plus`, five ordered keyframes, strict JSON Schema,
  bounded concurrency 2 and a content/model/prompt/frame cache. The cold
  benchmark made six calls; the warm benchmark made zero.
- C3: the existing deterministic visual-quality scorer ran on sixteen decoded
  official-loader frames per episode. All six sampled scores were 1.0.
- SAM3: official `facebook/sam3@3c879f39826c` was restored to the persistent
  cache. The checkpoint is 3,439,938,512 bytes with Git-LFS content SHA-256
  `6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a`.
  The Xet resolve ETag is a transport identifier and is not used as the file
  digest. A local synthetic CUDA forward produced a real mask.
- C2: three candidate keyframes per episode were batched for the entity prompt
  (`hands` for Human; `robot arms and hands` for Robot) and `manipulated object`.
  Masks are cached as NPZ. C2 compares real SAM3 entity masks to support discs
  around absolute EEF origins transformed through frozen Camera0 and stored-K.
  Robot EEFs are URDF-FK-derived; Human hands are measured. This is explicitly
  not a full URDF mesh silhouette, so the low IoUs are evidence for review, not
  a calibrated pass threshold.

Human C2 scores are 0.0185, 0.0176 and 0.0202; coverage is 3/8. Robot C2 scores
are 0.0000, 0.0122 and 0.0132; coverage is 4/8 because existing S2 is also
available. All final decisions remain `review`. S1/S3 remain N/A because a 3+3
pilot cannot fit a reference distribution. Robot S4 remains N/A without an
independent logged EEF, and S5 remains N/A without a reviewed expected
orientation/threshold authority.

## Timing

| Source/cache | Decode | Cheap rules | C1 | C3 | SAM3 | Sidecar | Audit | Total before audit | ep/h | video-h/h | API calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Human cold | 5.922s | 0.007s | 21.171s | 0.710s | 21.003s | 0.070s | 0.079s | 48.923s | 220.8 | 1.62 | 3 |
| Human warm | 5.394s | 0.080s | 0.249s | 0.744s | 0.054s | 0.135s | 0.078s | 6.701s | 1611.8 | 11.84 | 0 |
| Robot cold | 5.193s | 0.012s | 22.178s | 0.381s | 19.756s | 0.154s | 0.062s | 47.696s | 226.4 | 3.72 | 3 |
| Robot warm | 4.839s | 0.015s | 0.185s | 0.405s | 0.066s | 0.045s | 0.097s | 5.577s | 1936.4 | 31.85 | 0 |

With Human and Robot on two GPUs, the measured parallel aggregate is 441.5
episodes/hour cold and 3223.6 episodes/hour warm; the corresponding mixed
video throughput is 5.25 and 38.35 video-hours/hour. A sequential single-worker
projection is 223.6 and 1759.2 episodes/hour. These are sparse-keyframe pilot
rates, not full-frame segmentation rates.

The deterministic benchmark report includes per-1000-episode and
per-100-video-hour formulas. No full corpus was enumerated, so no unsupported
dataset-total claim is made. Before approving scale, multiply those unit costs
by an independently approved train-only episode/video-hour inventory.

## One-time versus steady state

The locked recovery reused 3,088,399,014 verified bytes and fetched
351,539,498 bytes. Exact remote-transfer wall time was not retained across the
initial fail-closed Xet-ETag/LFS-SHA diagnosis; it is intentionally not
invented. A separate first model-load plus synthetic forward took 32.299s.
Steady-state timings are the warm rows above: zero API calls, 3/3 C1 cache hits
and 18/18 SAM3 mask cache hits per source.

Outputs are under
`.rds-artifacts/live-quality-fast-benchmark-20260728/{human,robot}-{cold,warm}`;
`benchmark.json` contains the machine-readable summary. Every audit passed with
no non-JSON sidecar payload and no forbidden sensitive field.
