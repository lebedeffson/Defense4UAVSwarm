# Three-Frame Trust Story Summary

method: `geometry_dynamic_adaptive_balanced`
sequence: `uav0000268_05773_v`
frames: `2, 3, 4`

Available features shown: detector confidence `c_i`, temporal/geometric proxy `k_i`, `Q_i=min(c_i,k_i)`, temporal age, decision, limiting feature.
VisDrone is single-camera; no inter-agent consistency is shown or claimed.
Example selection: a tracklet with at least three frames and visible accepted/rejected trust decisions.
Label readability: generated as a wide three-panel figure; use in supplementary if page space is tight.
Ready for article: no; the selected transition is useful diagnostically but the accepted endpoint is not matched as TP.
Ready for supplementary: yes.

```text
 frame_id                 det_id  confidence      c_i      k_i      Q_i  temporal_age decision              reason  eval_is_tp
        2 uav0000268_05773_v_2_1    0.385665 0.385665 0.350000 0.350000             0 rejected    temporal_support       False
        3 uav0000268_05773_v_3_1    0.465960 0.465960 0.905158 0.465960             1  delayed    temporal_support        True
        4 uav0000268_05773_v_4_1    0.598019 0.598019 0.717341 0.598019             2 accepted detector_confidence       False
```
