# Defense4UAVSwarm: прозрачное описание исследовательского фреймворка

## 1. Назначение

Defense4UAVSwarm — экспериментальный фреймворк для проверки устойчивости нейросетевого восприятия беспилотных транспортных систем к FGSM-возмущениям.

Идея не в обучении новой сети с нуля, а в постдетекционном защитном слое поверх готового detector + tracker. Слой использует confidence детектора и кинематическую согласованность объекта во времени, объединяет признаки через T-нормы и фильтрует подозрительные детекции.

В первой версии XAI отключен намеренно: сначала нужно доказать, что базовая защита `confidence + kinematics + T-норма` дает измеримый выигрыш относительно S1 и `S_naive`.

## 2. Сценарии

### S0 — чистый режим

Кадры не атакуются. Detector + tracker работают штатно. S0 нужен для baseline, подбора `tau_conf*`, подбора `tau_Q*` и сравнения с атакованными режимами.

### S1 — FGSM без защиты

На вход добавляется FGSM-возмущение, фильтрация не применяется. S1 показывает уязвимость базового detector + tracker.

### S_naive — FGSM + confidence threshold

Простая защита: отбрасываются детекции ниже confidence-порога. Порог не фиксируется как `0.5`, а выбирается на `S0-val` по максимуму F1:

```text
confidence in {0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80}
```

### S2 — FGSM + T-нормы без XAI

Для каждой детекции считаются:

```text
c_i — confidence
k_i — kinematic consistency
s_i — inter-agent consistency
x_i — XAI consistency
```

В первой версии:

```text
s_i = 1
x_i = 1
```

Фактически проверяется связка:

```text
confidence + kinematics + T-норма
```

## 3. Почему s_i = 1 и x_i = 1

Для T-норм значение `1` является нейтральным элементом. Поэтому при `s_i=1` и `x_i=1`:

```text
T_min = min(c_i, k_i)
T_prod = c_i * k_i
T_Lukasiewicz = max(0, c_i + k_i - 1)
```

Это честно фиксирует ограничение VisDrone: датасет не является полноценной многoагентной кооперативной сценой, а XAI пока не проверяется.

## 4. Кинематическая согласованность

Используется модель постоянной скорости:

```text
p_hat(t) = p(t-1) + v(t-1)
```

Затем считается расстояние между прогнозируемым центром и текущей детекцией:

```text
k_i = exp(-d_i / alpha)
```

Если FGSM создает ложную или резко смещенную детекцию, confidence может быть высоким, но `k_i` будет низким.

## 5. T-нормы

Defense4UAVSwarm считает совместное доверие:

```text
Q_i = T(c_i, k_i, s_i, x_i)
```

Реализованы:

```text
T_min = min(c_i, k_i, s_i, x_i)
T_prod = c_i * k_i * s_i * x_i
T_Lukasiewicz = max(0, c_i + k_i + s_i + x_i - 3)
```

`T_min` — самый жесткий оператор. `T_prod` мягче, но штрафует совместное снижение признаков. `T_Lukasiewicz` дает строгую конъюнкцию с отсечением слабых комбинаций.

## 6. Подбор порогов

Все пороги выбираются только на `S0-val`.

Для `S_naive`:

```text
tau_conf* in {0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80}
```

Для `S2`:

```text
tau_Q* in {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}
```

Критерий — максимум F1. Если F1 отличается меньше чем на `0.01`, выбирается меньший порог.

## 7. Почему XAI отключен

XAI не запускается в первой версии, потому что Grad-CAM для YOLOv8 сложнее, увеличивает latency, а без рабочих S0-S2 невозможно измерить его дополнительный вклад.

В первой версии:

```text
x_i = 1
```

S3 открывается только после воспроизводимых S0-S2:

```text
S3 — FGSM + T-нормы + selective XAI
```

## 8. Research matrix

Один запуск на одной модели не доказывает устойчивость. Поэтому фреймворк проверяет:

```text
models
eps levels
scenarios
T-norms
class groups
detection/tracking tasks
```

Минимум моделей:

```text
YOLOv8n
YOLOv8s
```

Расширенно:

```text
YOLOv8n
YOLOv8s
YOLOv8m
```

FGSM:

```text
eps in {0.002, 0.004, 0.008, 0.016}
```

Class groups:

```text
all
vru
vehicles
```

## 9. Основной файл

Главный результат:

```text
outputs/results/research_matrix.csv
```

Обязательные поля:

```text
task
dataset_subset
model_name
scenario
eps
class_group
t_norm
tau
tau_conf
precision
recall
F1
mAP
MOTA
IDF1
IDSW
FP
FN
ASR
track_breaks
latency_ms
num_sequences
num_frames
```

## 10. Абляция

Файл:

```text
outputs/results/ablation_summary.csv
```

Проверяются варианты:

```text
S1 — FGSM без защиты
S_naive — confidence threshold
S2_c — только confidence
S2_ck — confidence + kinematics
S2_ck_tmin — c+k через T_min
S2_ck_tprod — c+k через T_prod
S2_ck_tluk — c+k через T_Lukasiewicz
```

## 11. Успешный результат

Метод считается рабочим, если:

1. S1 деградирует относительно S0.
2. S_naive частично снижает деградацию.
3. S2 дает выигрыш относительно S1.
4. S2 дает выигрыш относительно S_naive хотя бы по части метрик.
5. S2 снижает FP или ASR без критического роста FN.
6. Эффект виден не только на одной модели.
7. Эффект виден хотя бы на VRU или vehicles.
8. Результаты воспроизводятся через `metadata.json`, configs и CSV.

Если S2 улучшает FP, ASR и IDSW, но немного ухудшает recall, это допустимо: фильтр работает как консервативная защита.

## 12. Что сдавать после research run

```text
outputs/results/research_matrix.csv
outputs/results/ablation_summary.csv
outputs/results/threshold_selection.csv
outputs/results/summary_metrics.csv
outputs/results/metadata.json
outputs/figures/f1_vs_eps_by_model.png
outputs/figures/asr_vs_eps_by_model.png
outputs/figures/class_group_comparison.png
outputs/figures/tnorm_comparison.png
outputs/figures/ablation_variants.png
```

Короткий отчет:

```text
1. Какие модели реально запущены?
2. Какой датасет использован: VisDrone2019-VID или MOT?
3. Сколько sequence и кадров обработано?
4. Какой FGSM loss использован: full, proxy или class_only?
5. Есть ли падение S1 относительно S0?
6. Дает ли S2 выигрыш относительно S1?
7. Дает ли S2 выигрыш относительно S_naive?
8. Какая T-норма лучшая по F1?
9. Какая T-норма лучшая по ASR?
10. Где метод работает лучше: VRU или vehicles?
11. Как меняется эффект при росте eps?
12. Какова средняя latency?
```

## 13. Главное правило

Пока нет рабочих S0-S2, XAI не трогать.

Текущий цикл:

```text
clean baseline -> attack -> naive defense -> T-norm defense -> comparison by models/classes
```

## 14. Audit Note

Первые реальные прогоны на COCO-pretrained `yolov8n.pt/yolov8s.pt` показали ограничение постановки: baseline recall на VisDrone низкий даже в class-agnostic режиме. Это не доказывает слабость T-норм; это означает, что нужен detector, адаптированный к VisDrone, либо отдельный class-agnostic протокол.

Corrected-протокол фиксирует:

```text
model.conf = 0.05
prediction_class_filter = person,bicycle,car,motorcycle,bus,truck
tau_Q grid = 0.05..0.50
tracking metrics включены для class_group=all
hard и soft filtering сравниваются отдельно
```

Оставшиеся ограничения:

```text
FGSM preprocessing пока resize-based, а Ultralytics inference использует letterbox
COCO->VisDrone remap неполный
fine-tune на VisDrone нужен для сильного baseline
```

## 15. DET-only Auxiliary Task

`VisDrone2019-DET` не заменяет `VID/MOT` для трекинговой части: в DET нет нормальных видеопоследовательностей и `track_id`, поэтому `MOTA`, `IDF1`, `IDSW`, `track_breaks` и track-level ASR не считаются.

DET используется только как вспомогательная проверка устойчивости детектора на статических изображениях:

```text
task = det
kinematics_enabled = false
tracking_enabled = false
k_i = s_i = x_i = 1
scenario S2_conf = detection-only confidence T-норма
```

Для DET считаются только detection-метрики (`precision`, `recall`, `F1`, `mAP`, `FP`, `FN`, `latency_ms`) и отдельный `ASR_det`: доля изображений, где после FGSM появился пропуск clean-detected объекта, выросло число FP или F1 по изображению упал ниже заданного delta-порога.

Threshold selection разделяет нижний порог модели и порог пост-фильтра:

```text
model_conf = detector inference floor
S_naive confidence grid = 0.20 ... 0.80
S2_conf tau_Q_det grid = 0.05, 0.10, 0.20, 0.30, 0.40, 0.50
```

Если несколько порогов дают одинаковый F1 или отличаются меньше чем на `0.01`, выбирается меньший порог. Это правило применяется одинаково для `S_naive`, `S2` и `S2_conf`.

COCO-to-VisDrone class mapping фиксируется в `configs/class_mapping_coco_to_visdrone.yaml`. Для COCO-pretrained YOLO используются только классы `person`, `bicycle`, `car`, `truck`, `bus`, `motorcycle`; `tricycle` и `awning-tricycle` остаются unmapped. Class groups `all`, `vru` и `vehicles` интерпретируются после этого mapping/filtering.

Чтобы объяснить, почему фильтр помогает или не помогает, DET-summary сохраняет `num_detections_before_filter`, `num_detections_after_filter`, `num_rejected`, `rejection_rate`, а FGSM diagnostics сохраняют `mean_abs_perturbation`, `max_abs_perturbation` и `mean_gradient_norm`. Для sanity-check метрик формируется `metric_debug_sample.csv` с GT/TP/FP/FN по первым изображениям.

## 16. VID Calibration/Holdout Protocol

VID/MOT results use a fixed calibration/holdout protocol. Calibration sequences are used to select defense parameters; holdout sequences are used for final reporting. The split is recorded in `configs/vid_split.yaml` and copied into run metadata as `vid_split_file`, `tuning_split`, and `evaluation_split`.

The original clean-only threshold selection remains a baseline. The robust defense selection ranks candidates on attacked calibration data subject to a clean-quality constraint:

```text
clean_F1(S2) >= clean_F1(S0) - 0.02
```

The preferred score is:

```text
robust_score = 0.5 * F1_attack + 0.3 * IDF1_attack - 0.2 * ASR_track
```

If IDF1 is not computed during a fast sweep, the fallback score is:

```text
robust_score = 0.7 * F1_attack - 0.3 * ASR_track
```

The kinematic term is evaluated as three variants:

```text
k_center = exp(-d_center / alpha)
k_iou = IoU(bbox_predicted, bbox_detected)
k_combined = sqrt(k_center * k_iou)
```

where `bbox_predicted` is obtained by constant-velocity extrapolation of the previous tracked box. `alpha = alpha_scale * alpha_base`, with `alpha_base` derived from the tracked bbox scale. Temporal smoothing is optional:

```text
Q_smooth(t) = beta * Q(t) + (1 - beta) * Q_smooth(t-1)
```

Both `hard_filter` and `soft_reweight` are supported. Hard filtering accepts detections with `Q_i >= tau_Q`; soft reweighting uses `confidence_new = confidence * Q_i` and then applies the threshold.

### Pre-association kinematics

The defensive kinematic feature must be computed before accepting the current detection into the defense state. The defense state is independent of the tracker state already updated by ByteTrack. For each frame, active defense tracks are extrapolated with constant velocity, detections are matched to predicted tracks, and `k_i` is computed from the residual before the state is updated.

No-op candidates are not valid selected defenses. A candidate must satisfy:

```text
clean_F1_drop <= 0.02
rejection_rate_attack >= 0.01
```

If no candidate satisfies both constraints, the run writes `selection_status: not_selected` and holdout should not be run.

### Recall-aware v1.1 selection

After pre-association kinematics makes `k_i` non-neutral, calibration uses a recall-aware objective to avoid selecting an overly aggressive filter:

```text
robust_score =
 0.50 * F1_attack
-0.25 * ASR_track
-0.15 * FN_rate_attack
-0.10 * clean_F1_drop
```

The primary constraints are:

```text
clean_F1_drop <= 0.02
recall_drop_vs_S1 <= 0.03
rejection_rate_attack >= 0.01
```

### Track-aware v1.2 selection

v1.2 does not apply one global reject rule to every detection. It separates:

```text
confirmed_existing
tentative_existing
new_candidate
unmatched_detection
```

For `confirmed_existing`, low-Q detections are kept and optionally downweighted. For `tentative_existing`, rejection is delayed by `reject_patience`. For `new_candidate/unmatched_detection`, `tau_new` can suppress new false tracks.

The selection score is:

```text
robust_score =
 0.30 * F1_attack
+0.25 * IDF1_attack
-0.15 * ASR_any
-0.15 * IDSW_rate
-0.10 * track_break_rate
-0.05 * clean_F1_drop
```

Holdout is allowed only if calibration finds a candidate satisfying:

```text
clean_F1_drop <= 0.02
recall_drop_vs_S1 <= 0.02
IDSW_delta_vs_S1 <= 0
track_break_delta_vs_S1 <= 0
rejection_rate_attack >= 0.005
```

and at least one improvement in FP, failure intensity, or IDF1. Otherwise `selection_status: not_selected` means holdout is blocked.

If no candidate satisfies the primary clean constraint, a fallback candidate may be marked with `selection_status = fallback_clean_003` when `clean_F1_drop <= 0.03`. Holdout is still gated on the error trade-off: reducing FP alone is not enough if FN, IDF1, IDSW, track breaks, or total error intensity degrade.

`soft_reweight` can apply a confidence floor:

```text
confidence_new = max(soft_conf_floor, confidence * Q_i)
```

`delayed_hard_filter` rejects only after `Q_i < tau_Q` for `reject_patience` consecutive accepted defense-state updates.
