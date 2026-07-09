# Confidence Threshold Baseline Claim-Safe Notes

Protocol:
- detector: `yolov8n`
- matching_mode: `coarse_class`
- ignore_policy: `exclude_ignored`
- iou_threshold: `0.5`
- detector_conf_threshold before evaluation: `0.1`

Threshold sweep:

```text
 threshold       F1  false_new_tracks  precision   recall
       0.1 0.389001              9974   0.500386 0.318175
       0.2 0.354442              3745   0.680433 0.239635
       0.3 0.302046              1573   0.803034 0.186004
       0.4 0.246039               646   0.880734 0.142992
       0.5 0.187781               280   0.924645 0.104502
       0.6 0.129029                97   0.954320 0.069192
       0.7 0.071154                34   0.970304 0.036931
```

Best confidence threshold by F1: `0.1` with F1=0.389001, false_new=9974.

Questions:

1. Can a simple confidence threshold reach the trust-layer false_new level (4250)?
Yes, at threshold 0.2 or above; best such F1=0.354442 (delta vs trust -0.026793).
2. What happens to F1 at that threshold?
F1 drops to 0.354442 at the best threshold that reaches the trust false_new level.
3. Is there a point where confidence threshold is better than the trust layer?
No under the reported thresholds.
4. Is there a point where trust layer is better at close F1?
Closest-F1 threshold is 0.1 with F1=0.389001, false_new=9974; trust has F1=0.381235, false_new=4250.
5. Safe article claim:
A simple confidence threshold can reduce false-new tracks, but reaching the trust false-new range costs additional F1 in this sweep.
