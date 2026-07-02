# Defense4UAVSwarm

Минимальный pipeline по ТЗ v0.6. Приоритет: сначала S0/S1/S_naive/S2 и первые числа, XAI только вторым этапом.

Прозрачное описание логики фреймворка и research matrix: [docs/research_framework.md](docs/research_framework.md).

## Установка

```bash
pip install -r requirements.txt
```

Для локальной разработки пакета: `pip install -e .`

## Датасет

Основной набор: VisDrone2019-VID. Fallback: VisDrone2019-MOT.

Скачать train/val с официальной страницы VisDrone:
https://github.com/VisDrone/VisDrone-Dataset

Прямые Google Drive ссылки с официального README:

```text
VID train: https://drive.google.com/file/d/1NSNapZQHar22OYzQYuXCugA3QlMndzvw/view?usp=sharing
VID val:   https://drive.google.com/file/d/1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B/view?usp=sharing
MOT train: https://drive.google.com/file/d/1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh/view?usp=sharing
MOT val:   https://drive.google.com/file/d/1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu/view?usp=sharing
```

Положить архивы в `data/visdrone/` и распаковать так, чтобы внутри были каталоги вида:

```text
data/visdrone/VisDrone2019-VID-val/
  sequences/
  annotations/
```

Проверка чтения кадров, bbox, track_id/classes и 20-30 GT-визуализаций:

```bash
python scripts/check_dataset.py --root data/visdrone
python scripts/prepare_visdrone.py --config configs/default.yaml --verify
```

Если официального val нет:

```bash
python scripts/prepare_visdrone.py --config configs/default.yaml --custom-split --verify
```

Split будет записан в `configs/visdrone_split.yaml`, metadata в `outputs/results/metadata.json`.

## Baseline S0

Smoke test без полного VisDrone:

```bash
python scripts/run_pipeline.py --config configs/default.yaml --dry-run
```

Первый реальный запуск на одной sequence:

```bash
python scripts/run_pipeline.py --config configs/default.yaml --scenario s0 --limit-sequences 1
```

Первый контрольный результат:

```text
outputs/results/s0_baseline.csv
outputs/results/summary_metrics.csv
outputs/results/metadata.json
```

## FGSM S1

В текущем каркасе FGSM включен для eps:

```text
0.002, 0.004, 0.008, 0.016
```

Loss зафиксирован как `class_only` fallback в `configs/default.yaml` и `metadata.json`. Полный YOLO loss надо подключать отдельно, если удается за 2 рабочих дня.

Запуск S1 на одной sequence и одном eps:

```bash
python scripts/run_pipeline.py --config configs/default.yaml --scenario s1 --eps 0.004 --limit-sequences 1
```

## S_naive

`tau_conf*` выбирается автоматически на S0-val по F1 из сетки:

```text
0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80
```

Tie-break: при разнице F1 меньше `0.01` выбирается меньший порог.

Важно: inference confidence в `configs/default.yaml` намеренно ниже сетки подбора:

```yaml
model:
  conf: 0.05
```

Иначе `S_naive` вырождается в `S1`, если detector уже отрезал все ниже `0.20`.

## S2 T-нормы

Реализованы:

```text
T_min
T_prod
T_Lukasiewicz
```

`tau_Q*` выбирается на S0-val отдельно для каждой T-нормы из:

```text
0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50
```

`s_i=1`, `x_i=1`, что записывается как `neutral_single_camera`.

Для COCO-pretrained YOLO перед оценкой оставляются только классы, сопоставимые с VisDrone:

```text
person, bicycle, car, motorcycle, bus, truck
```

Остальные COCO-классы отбрасываются до метрик, чтобы не смешивать `train/cell phone/traffic light` и другие out-of-domain FP с задачей VisDrone.

Запуск S2 на одной sequence и одном eps:

```bash
python scripts/run_pipeline.py --config configs/default.yaml --scenario s2 --eps 0.004 --limit-sequences 1
```

Проверка unit test формул T-норм:

```bash
pytest tests/test_tnorms.py
```

## Пересчитать таблицы и графики

```bash
python scripts/run_pipeline.py --config configs/default.yaml --stage s0_s2
python scripts/make_figures.py --summary outputs/results/summary_metrics.csv
```

## Research Matrix

Стартовый неполный прогон для проверки обобщаемости:

```bash
python scripts/run_experiment_matrix.py \
  --config configs/default.yaml \
  --models yolov8n yolov8s \
  --tasks vid \
  --eps 0.004 0.008 \
  --scenarios s0 s1 s_naive s2 \
  --class-groups all vru vehicles \
  --limit-sequences 3
```

Полный перебор запускать только после проверки `research_matrix.csv`.

Corrected audit hard/soft:

```bash
python scripts/run_experiment_matrix.py \
  --config configs/default.yaml \
  --models yolov8n yolov8s \
  --tasks vid \
  --eps 0.016 0.032 \
  --scenarios s0 s1 s_naive s2 \
  --class-groups all vru vehicles \
  --fgsm-loss class_only \
  --sequence-list configs/selected_sequences.yaml \
  --class-agnostic-eval true \
  --filter-mode hard \
  --output-dir outputs/results/corrected_hard \
  --skip-ablation
```

Для soft заменить `--filter-mode soft` и `--output-dir outputs/results/corrected_soft`.

## Task A: DET-only robustness

`VisDrone2019-DET` используется только как вспомогательная detection-only ветка. Он не заменяет `VisDrone2019-VID/MOT` для трекинговых метрик.

Поддерживаются две структуры:

```text
VisDrone2019-DET-val/images
VisDrone2019-DET-val/annotations
```

или Ultralytics:

```text
VisDrone/images/val
VisDrone/labels/val
```

В `task=det` кинематика и трекинг отключены:

```text
kinematics_enabled = false
tracking_enabled = false
k_i = s_i = x_i = 1
MOTA/IDF1/IDSW/track_breaks = null
```

Сценарий `S2` записывается как `S2_conf`, то есть detection-only confidence T-норма. Для DET дополнительно пишется `ASR_det`.

В DET-режиме используется отдельная сетка порога `tau_Q_det`:

```text
0.05, 0.10, 0.20, 0.30, 0.40, 0.50
```

Это не `model_conf`; `model_conf` задает нижний порог детектора, а `tau_Q_det` выбирается как пост-фильтр. При почти одинаковом F1 действует единый tie-break: выбирается меньший порог.

COCO-pretrained YOLO сопоставляется с VisDrone через явный файл:

```text
configs/class_mapping_coco_to_visdrone.yaml
```

Метрики считаются после этого mapping/filtering. `tricycle` и `awning-tricycle` остаются unmapped для COCO-моделей.

Для проверки, работает ли фильтрация, в summary пишутся:

```text
num_detections_before_filter
num_detections_after_filter
num_rejected
rejection_rate
```

FGSM diagnostics сохраняют `mean_abs_perturbation`, `max_abs_perturbation` и `mean_gradient_norm`.

Пример:

```bash
python scripts/run_experiment_matrix.py \
  --config configs/default.yaml \
  --models yolov8n yolov8s \
  --tasks det \
  --eps 0.004 0.008 \
  --scenarios s0 s1 s_naive s2 \
  --class-groups all vru vehicles \
  --limit-images 200 \
  --output-dir outputs/results/det_only
```

Выходы:

```text
outputs/results/research_matrix.csv
outputs/results/ablation_summary.csv
outputs/results/threshold_selection.csv
outputs/results/metric_debug_sample.csv
outputs/figures/f1_vs_eps_by_model.png
outputs/figures/asr_vs_eps_by_model.png
outputs/figures/class_group_comparison.png
outputs/figures/ablation_variants.png
outputs/figures/fgsm_examples/
```

## VID calibration protocol

`VisDrone2019-VID-val` is split into calibration and holdout sequences in:

```text
configs/vid_split.yaml
```

Calibration is used for defense parameter selection only. Holdout is reserved for final reporting. The robust selection objective keeps clean degradation bounded:

```text
clean_F1(S2) >= clean_F1(S0) - 0.02
```

and then ranks candidates by attacked robustness. v1.2 uses a tracking-aware score:

```text
robust_score =
 0.30 * F1_attack
+0.25 * IDF1_attack
-0.15 * ASR_any
-0.15 * IDSW_rate
-0.10 * track_break_rate
-0.05 * clean_F1_drop
```

The VID kinematic feature is computed in pre-association mode: a separate defense track state predicts the next bbox before the current detection is accepted or rejected. A no-op candidate cannot be selected; calibration requires `clean_F1_drop <= 0.02`, `recall_drop_vs_S1 <= 0.02`, `IDSW_delta_vs_S1 <= 0`, `track_break_delta_vs_S1 <= 0`, and `rejection_rate_attack >= 0.005`. If no candidate passes the tracking constraints, `selected_defense_params.yaml` is written with `selection_status: not_selected`; holdout must not be run.

The v1.2 calibration pass supports `track_aware` and `new_track_suppression`. Confirmed tracks are not hard-dropped; new or unmatched detections can be suppressed by `tau_new`. It writes `track_status_summary.csv`, `track_lifecycle_debug.csv`, `fp_source_summary.csv`, and `error_intensity_summary.csv`.

VID defense candidates compare `k_variant` (`center`, `iou`, `combined`), `alpha_scale`, `filter_mode` (`hard_filter`, `soft_reweight`), `beta`, T-norm, and `tau_Q`. A lightweight selection pass can be run after S0/S1 CSVs exist:

```bash
python scripts/select_vid_defense.py \
  --config configs/default.yaml \
  --split-config configs/vid_split.yaml \
  --split calibration \
  --results outputs/results/vid_calibration \
  --model yolov8n \
  --eps 0.008 \
  --k-variants robust_min gate \
  --filter-modes new_track_suppression track_aware \
  --alpha-scales 0.30 0.50 \
  --gamma-assoc 0.05 \
  --tau-existing-grid 0.10 0.15 \
  --tau-new-grid 0.20 0.25 0.30 \
  --betas none 0.7 \
  --reject-patience 2
```

Таблицы для статьи:

```bash
python scripts/build_research_tables.py \
  --matrix outputs/results/research_matrix.csv \
  --ablation outputs/results/ablation_summary.csv
```

Будут созданы:

```text
outputs/results/table1_model_robustness.csv
outputs/results/table2_class_robustness.csv
outputs/results/table3_tnorm_comparison.csv
outputs/results/table4_ablation_features.csv
outputs/results/table5_eps_sensitivity.csv
```

Основные файлы:

```text
outputs/results/s0_baseline.csv
outputs/results/s1_fgsm.csv
outputs/results/s_naive.csv
outputs/results/s2_tnorm.csv
outputs/results/threshold_selection.csv
outputs/results/summary_metrics.csv
outputs/results/metadata.json
outputs/figures/metrics_vs_eps.png
outputs/figures/t_norm_comparison.png
```

XAI/S3 намеренно не запускается до готовых S0-S2.
