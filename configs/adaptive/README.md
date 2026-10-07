# configs/adaptive/

JSON run configs for `adaptive_training.py`, which builds an adaptive (gated, low-rank) network,
warm-starts it from a baseline checkpoint, and trains it.

Every key is optional; anything omitted falls back to `SCENARIO_DEFAULTS` in
`../config_utils.py`. Unknown keys are rejected. The `arch` value must be a factory exported by
`arch/models_adaptive.py`. `resume` must point to a plain-backbone checkpoint produced by
`train_base.py`, not an adaptive one. Evaluating an existing adaptive checkpoint uses
`"cmd": "test"` instead.

