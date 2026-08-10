# F1 / False-New Trade-off Claim-Safe Notes

Points:

```text
                        label               family       F1  false_new_tracks  pareto
                    ByteTrack              tracker 0.453987            6838.0   False
   ByteTrack + geometry trust                trust 0.450787            4799.0   False
ByteTrack + adaptive balanced                trust 0.451021            4803.0   False
   ByteTrack + false-new-safe                trust 0.449439            4773.0   False
                           RF           supervised 0.470172            3440.0    True
                      OC-SORT              tracker 0.441120            2184.0    True
     OC-SORT + geometry trust                trust 0.432105            1726.0    True
                  conf >= 0.1 confidence_threshold 0.451579           11682.0   False
                  conf >= 0.2 confidence_threshold 0.435418            4581.0   False
                  conf >= 0.3 confidence_threshold 0.389417            2013.0   False
                  conf >= 0.4 confidence_threshold 0.335154             933.0    True
                  conf >= 0.5 confidence_threshold 0.276499             418.0    True
                  conf >= 0.6 confidence_threshold 0.215849             192.0    True
                  conf >= 0.7 confidence_threshold 0.148429              86.0    True
```

1. Does the trust layer Pareto-dominate ByteTrack?
No. It reduces false-new tracks but loses F1 relative to ByteTrack.
2. Does the trust layer dominate the confidence-threshold baseline?
No / partial; inspect the threshold points.
3. Best point for the article:
ByteTrack + geometry trust (F1=0.450787, false_new=4799).
4. Most conservative point:
ByteTrack + false-new-safe (F1=0.449439, false_new=4773).
5. Safe claim:
Trust shifts the operating point toward fewer false-new tracks at a small F1 cost compared with ByteTrack and is not matched by the closest-F1 confidence threshold.
6. Forbidden claim:
Do not claim universal Pareto dominance or simultaneous F1/false-new improvement unless all plotted baselines support it.
