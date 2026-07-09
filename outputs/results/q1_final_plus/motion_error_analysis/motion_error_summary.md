# Motion Error Analysis Summary

Status: done with limitations.

Motion groups are computed from VisDrone GT object center displacement in pixels per frame.
TP/FN are grouped by matched GT object speed. FP/false-new tracks have no true object speed, so FP values are retained as unmatched candidate proxies.
This is a diagnostic proxy, not a stable motion benchmark.

GT objects with speed estimate: 758

```text
                        method speed_bin       F1    FP    FN  false_new_tracks
                     bytetrack      slow 0.447737 36503 59788              6838
                     bytetrack    medium 0.447297 36503 62386              6838
                     bytetrack      fast 0.461611 36503 55737              6838
geometry_dynamic_no_multiagent      slow 0.447594 30652 61491              4799
geometry_dynamic_no_multiagent    medium 0.446051 30652 64209              4799
geometry_dynamic_no_multiagent      fast 0.461382 30652 57517              4799
```
