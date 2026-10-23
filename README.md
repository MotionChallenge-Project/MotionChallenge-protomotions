# Motion Challenge · protomotions

[protomotions](https://github.com/NVlabs/ProtoMotions) checkpoint adapter. G1 29DoF; LAFAN40 is the first suite.

## Install and run

```sh
git clone --recurse-submodules https://github.com/motion-challenge-project/motion-challenge-protomotions.git
cd motion-challenge-protomotions
uv sync
uv run evaluate.py
```

Reuses local `checkpoints/policy.onnx` or the external cache; missing artifacts are
fetched and hash-checked. Use `--weights PATH --data PATH --offline` for local files.
Results and traces are written outside Git. `--help` lists options.

The root `motion-challenge` submodule owns data, physics, metrics and verification.
This repository only adapts model inference; no `sim2real` dependency.

**Development adapter — author-inference review pending.**
[Provenance](provenance.json)

MuJoCo 3.17 batch is the default. Use `--jobs 8` for parallel clips;
download and evaluation progress updates every 5 seconds.

Run all frames after tracking failure: `uv run evaluate.py --termination none`.
This records full-motion errors under a separate protocol; success still requires no failure.

## Checkpoint source

[NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions/blob/f69fdcaabff31846b4d2247207417b57dc788828/README.md) · [unified_pipeline.onnx](https://github.com/NVlabs/ProtoMotions/blob/f69fdcaabff31846b4d2247207417b57dc788828/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx). Author-released checkpoint.

The adapter pins this published artifact by SHA256. Artifact identity is separate from inference parity.
