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

## S2 T-нормы

Реализованы:

```text
T_min
T_prod
T_Lukasiewicz
```

`tau_Q*` выбирается на S0-val отдельно для каждой T-нормы из:

```text
0.30, 0.40, 0.50, 0.60
```

`s_i=1`, `x_i=1`, что записывается как `neutral_single_camera`.

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

Выходы:

```text
outputs/results/research_matrix.csv
outputs/results/ablation_summary.csv
outputs/results/threshold_selection.csv
outputs/figures/f1_vs_eps_by_model.png
outputs/figures/asr_vs_eps_by_model.png
outputs/figures/class_group_comparison.png
outputs/figures/ablation_variants.png
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
