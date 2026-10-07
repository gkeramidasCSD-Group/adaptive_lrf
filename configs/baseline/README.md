# configs/baseline/

JSON run configs for `train_base.py`, which trains a plain (non-adaptive) backbone.

Every key is optional; anything omitted falls back to `BASE_DEFAULTS` in
`../config_utils.py`. Unknown keys are rejected. The `arch` value must be a factory exported by
`arch/models_baseline.py`.


