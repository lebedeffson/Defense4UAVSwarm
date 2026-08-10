# Confidence Threshold Baseline Claim-Safe Notes

Protocol:
- detector: `yolov8s`
- matching_mode: `coarse_class`
- ignore_policy: `exclude_ignored`
- iou_threshold: `0.5`
- detector_conf_threshold before evaluation: `0.1`

Threshold sweep:

```text
 threshold       F1  false_new_tracks  precision   recall
       0.1 0.451579             11682   0.499692 0.411918
       0.2 0.435418              4581   0.647673 0.327945
       0.3 0.389417              2013   0.743005 0.263852
       0.4 0.335154               933   0.810378 0.211264
       0.5 0.276499               418   0.853381 0.164976
       0.6 0.215849               192   0.879754 0.123015
       0.7 0.148429                86   0.908232 0.080819
```

Best confidence threshold by F1: `0.1` with F1=0.451579, false_new=11682.

Questions:

1. Can a simple confidence threshold reach the trust-layer false_new level (4799)?
Yes, at threshold 0.2 or above; best such F1=0.435418 (delta vs trust -0.015369).
2. What happens to F1 at that threshold?
F1 drops to 0.435418 at the best threshold that reaches the trust false_new level.
3. Is there a point where confidence threshold is better than the trust layer?
No under the reported thresholds.
4. Is there a point where trust layer is better at close F1?
Closest-F1 threshold is 0.1 with F1=0.451579, false_new=11682; trust has F1=0.450787, false_new=4799.
5. Safe article claim:
A simple confidence threshold can reduce false-new tracks, but reaching the trust false-new range costs additional F1 in this sweep.
