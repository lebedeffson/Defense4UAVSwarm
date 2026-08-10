# F1 / False-New Trade-off Claim-Safe Notes

Points:

```text
                        label               family       F1  false_new_tracks  pareto
                    ByteTrack              tracker 0.387191            6044.0   False
   ByteTrack + geometry trust                trust 0.381235            4250.0   False
ByteTrack + adaptive balanced                trust 0.381303            4251.0   False
   ByteTrack + false-new-safe                trust 0.380181            4245.0   False
                           RF           supervised 0.387684            3146.0    True
                      OC-SORT              tracker 0.370785            1866.0    True
     OC-SORT + geometry trust                trust 0.359343            1445.0    True
                  conf >= 0.1 confidence_threshold 0.389001            9974.0    True
                  conf >= 0.2 confidence_threshold 0.354442            3745.0   False
                  conf >= 0.3 confidence_threshold 0.302046            1573.0   False
                  conf >= 0.4 confidence_threshold 0.246039             646.0    True
                  conf >= 0.5 confidence_threshold 0.187781             280.0    True
                  conf >= 0.6 confidence_threshold 0.129029              97.0    True
                  conf >= 0.7 confidence_threshold 0.071154              34.0    True
```

1. Does the trust layer Pareto-dominate ByteTrack?
No. It reduces false-new tracks but loses F1 relative to ByteTrack.
2. Does the trust layer dominate the confidence-threshold baseline?
No / partial; inspect the threshold points.
3. Best point for the article:
ByteTrack + geometry trust (F1=0.381235, false_new=4250).
4. Most conservative point:
ByteTrack + false-new-safe (F1=0.380181, false_new=4245).
5. Safe claim:
Trust shifts the operating point toward fewer false-new tracks at a small F1 cost compared with ByteTrack and is not matched by the closest-F1 confidence threshold.
6. Forbidden claim:
Do not claim universal Pareto dominance or simultaneous F1/false-new improvement unless all plotted baselines support it.
