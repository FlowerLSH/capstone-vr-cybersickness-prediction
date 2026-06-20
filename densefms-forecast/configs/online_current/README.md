# Online-Current Configs

Use `selected_fds_static4.yaml` as the default configuration for current-FMS tracking work.

This selected path keeps the rapid-rise risk head as an auxiliary training target (`risk_head_enabled: true`, `risk_loss_weight: 0.25`). Recent ablations showed that removing the risk head or setting its loss weight to zero hurts validation MAE and trajectory metrics.

The older `configs/online_fms_current_tracker_*.yaml` files are kept for reproducibility and ablations, but new experiments should branch from this selected config unless the goal explicitly says otherwise.
