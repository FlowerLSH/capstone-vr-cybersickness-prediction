# DenseFMS Future Forecasting 2x2 실험 로그

작성일: 2026-04-29

후속 수정 메모: DenseFMS 논문 정의상 FMS는 `0~20` 범위의 절대 척도이므로, 이후 코드에서는 FMS normalization을 split별 관측 min/max가 아니라 고정 `0~20`으로 수행한다. 이 로그의 2x2 run checkpoint도 해당 fixed split train set의 관측 min/max가 `0.0/20.0`이어서 수치상 같은 scale로 학습되었지만, 이후 split에서도 같은 규칙이 보장되도록 코드를 고정 척도로 명시했다.

## 목적

DenseFMS future FMS forecasting 코드베이스에서 고정 split 기반 2x2 실험을 실행하고, 학습 가능성, leakage 방지, static feature 사용 여부, raw trend loss 효과를 확인했다.

실험 조건은 다음 네 가지다.

| 조건 | static feature | loss mode | trend weight |
| --- | --- | --- | --- |
| no_static + level_only | 사용 안 함 | `level_only` | 0.0 |
| no_static + level_trend_raw | 사용 안 함 | `level_trend_raw` | 0.1 |
| static + level_only | age + gender | `level_only` | 0.0 |
| static + level_trend_raw | age + gender | `level_trend_raw` | 0.1 |

## 연구 설정

- Dataset path: `./DenseFMS/Dataset`
- Sampling interval: 0.5 seconds
- Calibration length: 30 seconds = 60 steps
- Recent window: 10 seconds = 20 steps
- Forecast horizon: 5 seconds = 10 steps
- Prediction target: current time `t` 기준 `FMS[t + 10]`
- Calibration input: first 30 seconds head motion + FMS
- Post-calibration input: head motion stream only
- Static features: optional age/gender covariates
- Default model output: future FMS sequence
- Legacy NOW/DELTA/gate multi-head loss: 기본 학습 objective에서 사용하지 않음

## 데이터 점검 결과

Dataset inspection command:

```bash
python scripts/inspect_densefms.py --data_dir ./DenseFMS/Dataset
```

점검 결과:

- CSV files: 428
- Total rows: 204,334
- Time column: `timestamp`
- FMS column: `fms`
- Head motion columns:
  - `acc_x`
  - `acc_y`
  - `acc_z`
  - `angular_velocity_x`
  - `angular_velocity_y`
  - `angular_velocity_z`
- Participant/session columns: CSV 내부에는 없음
- Participant ID: filename의 `PA###` 패턴에서 추론
- Static columns:
  - age: `age`
  - gender: `gender`
- Gender categories observed: `female`, `male`
- Static coverage: valid 428 sessions 모두 age/gender 사용 가능

## Split

Reusable split file:

```text
./artifacts/densefms_split_seed42.json
```

Split summary:

| split | sessions |
| --- | ---: |
| train | 316 |
| validation | 60 |
| test | 52 |

Participant-wise split을 사용했고 train/validation/test participant overlap은 없었다.

## Leakage 방지 확인

다음 조건을 확인했다.

- 30초 calibration 이후 ground-truth FMS는 model input으로 사용하지 않음
- 30초 이후 FMS는 loss/evaluation target으로만 사용
- target shift는 `horizon_steps = 10`
- recent window는 현재 시점 `t`까지의 head motion만 포함
- future head motion은 input으로 사용하지 않음
- head/FMS normalization은 train split scaler만 사용
- static age normalization은 train split mean/std만 사용
- gender는 `male`, `female`, `unknown` one-hot encoding 사용
- 네 실험 모두 동일 split file과 동일 428-session subset 사용

## 실행한 검증

Sanity tests:

```bash
python scripts/run_densefms_sanity_tests.py
```

통과한 항목:

- loss equivalence: `level_trend_raw` with `trend_weight=0.0` equals `level_only`
- raw trend loss correctness
- padding/valid mask correctness
- no legacy multi-head loss in default objective
- static off compatibility
- static on requirement and shape check
- static scaler/gender encoding
- 2x2 config loading
- leakage-related forward signature check

Smoke training:

```bash
python -m src.densefms_forecast.train --data_dir ./DenseFMS/Dataset --config configs/coff_lstm_no_static_level.yaml --model coff_lstm --loss_mode level_only --trend_weight 0.0 --split_file ./artifacts/densefms_split_seed42.json --run_name prefull_smoke_no_static_level --epochs 1 --limit_sessions 8
python -m src.densefms_forecast.train --data_dir ./DenseFMS/Dataset --config configs/coff_lstm_static_level.yaml --model coff_lstm --loss_mode level_only --trend_weight 0.0 --use_static --static_features age gender --split_file ./artifacts/densefms_split_seed42.json --run_name prefull_smoke_static_level --epochs 1 --limit_sessions 8
```

Smoke result:

| run | loss_total | val_MAE |
| --- | ---: | ---: |
| no_static + level_only | 0.04964 | 6.0804 |
| static + level_only | 0.04631 | 5.9265 |

## Full Training Commands

```bash
python -m src.densefms_forecast.train --data_dir ./DenseFMS/Dataset --config configs/coff_lstm_no_static_level.yaml --model coff_lstm --loss_mode level_only --trend_weight 0.0 --split_file ./artifacts/densefms_split_seed42.json --run_name coff_lstm_no_static_level

python -m src.densefms_forecast.train --data_dir ./DenseFMS/Dataset --config configs/coff_lstm_no_static_trend.yaml --model coff_lstm --loss_mode level_trend_raw --trend_weight 0.1 --split_file ./artifacts/densefms_split_seed42.json --run_name coff_lstm_no_static_trend_w0.1

python -m src.densefms_forecast.train --data_dir ./DenseFMS/Dataset --config configs/coff_lstm_static_level.yaml --model coff_lstm --loss_mode level_only --trend_weight 0.0 --use_static --static_features age gender --split_file ./artifacts/densefms_split_seed42.json --run_name coff_lstm_static_level

python -m src.densefms_forecast.train --data_dir ./DenseFMS/Dataset --config configs/coff_lstm_static_trend.yaml --model coff_lstm --loss_mode level_trend_raw --trend_weight 0.1 --use_static --static_features age gender --split_file ./artifacts/densefms_split_seed42.json --run_name coff_lstm_static_trend_w0.1
```

각 run의 `best.pt` checkpoint에 대해 test split evaluation을 수행했다.

## 2x2 결과

| run_name | use_static | loss_mode | best_epoch | val_MAE | test_MAE | test_RMSE | test_R2 | test_sMAPE | derivative_MAE | trend_sign_accuracy | high_FMS_precision | high_FMS_recall | high_FMS_F1 | high_FMS_FPR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `coff_lstm_no_static_level` | false | `level_only` | 30 | 2.7977 | 3.2300 | 4.2088 | 0.2503 | 60.9459 | 0.2165 | 0.0355 | 0.6318 | 0.7419 | 0.6824 | 0.3462 |
| `coff_lstm_no_static_trend_w0.1` | false | `level_trend_raw` | 8 | 2.8951 | 3.2294 | 4.2216 | 0.2458 | 59.4271 | 0.1956 | 0.0340 | 0.6128 | 0.7925 | 0.6912 | 0.4009 |
| `coff_lstm_static_level` | true | `level_only` | 12 | 2.7652 | 3.2912 | 4.2733 | 0.2272 | 62.6280 | 0.2322 | 0.0346 | 0.6296 | 0.6007 | 0.6148 | 0.2830 |
| `coff_lstm_static_trend_w0.1` | true | `level_trend_raw` | 12 | 2.8066 | 3.3126 | 4.2859 | 0.2226 | 62.1391 | 0.2205 | 0.0348 | 0.6232 | 0.6340 | 0.6286 | 0.3069 |

Detailed machine-readable outputs were saved under `runs/`, but `runs/` is intentionally ignored by git.

## 산출물 위치

Run directories:

- `runs/coff_lstm_no_static_level`
- `runs/coff_lstm_no_static_trend_w0.1`
- `runs/coff_lstm_static_level`
- `runs/coff_lstm_static_trend_w0.1`

Summary files:

- `runs/experiment_summary_2x2.csv`
- `runs/experiment_summary_2x2.md`

Best checkpoints:

- `runs/coff_lstm_no_static_level/best.pt`
- `runs/coff_lstm_no_static_trend_w0.1/best.pt`
- `runs/coff_lstm_static_level/best.pt`
- `runs/coff_lstm_static_trend_w0.1/best.pt`

These outputs are experiment artifacts and are not committed.

## 코드 변경 기록

`src/densefms_forecast/train.py`에 다음 저장 기능을 추가했다.

- `config_snapshot.json`
- `final.pt`
- `training_curves.csv`
- `training_curves.png`
- `val_predictions.csv`
- `test_predictions.csv`

이 변경은 향후 실험 재현성과 분석 편의를 위한 것이며, model/loss/data split logic 자체를 바꾸지 않는다.

## 해석

1. Static age/gender는 이번 fixed split에서는 test MAE를 개선하지 못했다.
   - `static_level`은 `no_static_level`보다 test MAE가 +0.0613 높았다.
   - `static_trend`는 `no_static_trend`보다 test MAE가 +0.0833 높았다.
2. Raw trend loss는 derivative MAE를 개선했다.
   - no_static: 0.2165 -> 0.1956
   - static: 0.2322 -> 0.2205
3. No-static 조건에서 trend loss는 test MAE를 거의 해치지 않았고 아주 미세하게 좋았다.
   - 3.2300 -> 3.2294
4. Static + trend는 trend metric은 개선했지만 level metric은 악화했다.
5. Trend loss는 high-FMS false positive rate를 증가시켰다.
   - no_static: 0.3462 -> 0.4009
   - static: 0.2830 -> 0.3069
6. 다음 GroupKFold 후보는 `coff_lstm_no_static_trend_w0.1`가 가장 적절해 보인다.

## 다음 단계

- `coff_lstm_no_static_trend_w0.1`를 우선 후보로 두고 participant GroupKFold validation을 수행한다.
- Static branch는 age/gender만으로는 이 split에서 명확한 이득이 없었으므로, subgroup별 성능과 overfitting 가능성을 추가 분석한다.
- Trend loss는 derivative metric 개선 효과가 있으나 false positive 증가가 관찰되므로 high-FMS threshold analysis를 함께 보고한다.
