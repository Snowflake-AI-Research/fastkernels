# Reference probes

Standalone reproductions of the twelve HF reference issues described in
[`../REFERENCES.md`](../REFERENCES.md). Each probe builds a small seeded model from
the HF configuration classes, triggers one issue, and writes a JSON result. No
prepared runs or model weights are needed.

## Setup

Use the repository's Python environment (torch 2.11 with CUDA, `ninja` on `PATH`)
and check out both Transformers revisions. The newer one needs newer hub packages,
installed to a separate directory:

```bash
git clone https://github.com/huggingface/transformers.git $HF_OLD
git -C $HF_OLD checkout da6c53e431f7c9ef0691239d4ce89b0f711ecad7
git clone https://github.com/huggingface/transformers.git $HF_NEW
git -C $HF_NEW checkout 89b6b17574892ec0770551537a3fe69d6886703e
pip install --no-deps --target $DEPS_NEW huggingface_hub==1.33.0 tokenizers==0.23.1 safetensors==0.8.0 kernels==0.17.0
```

The MRA and DeepSeek-V4 probes download kernels from the Hub on first use; do not
set `HF_HUB_OFFLINE=1`.

## Running

```bash
cd fastkernels/hf_coverage/reference_probes
export CUDA_VISIBLE_DEVICES=<idle gpu>
python run_all.py --hf-source $HF_OLD --out $RUNS/probes-old
python run_all.py --hf-source $HF_NEW --extra-path $DEPS_NEW --out $RUNS/probes-new
python probes.py sam_hq --hf-source $HF_OLD --out $RUNS/sam_hq.json
```

Each suite takes about 90 seconds on one GPU. `run_all.py` writes one JSON and log
per model plus `summary.json`, which compares each verdict with the expected
verdict for the detected revision. Each result records the Transformers revision
and path it used, a `verdict`, the key measurements, and any error text:

- `execution_error`: the reference raises.
- `wrong_computation`: it runs but produces a verifiably wrong value.
- `reference_selection_failure`: no loadable HF reference matches the published model.
- `ok`: the issue did not reproduce.
- `probe_error`: the probe failed before reaching the issue (exit status 1).

Last verified 2026-09-25 on a B200: all twelve verdicts matched on both revisions.
