# Feature Ablation Claim-Safe Notes

Controlled simulation feature ablation; no VisDrone inter-agent consistency is claimed.

```text
        feature_mode       F1  false_new_tracks
         c_geometric 0.809873             551.0
                 c_k 0.818374            1444.0
       c_k_geometric 0.809816             539.4
        c_k_temporal 0.901778            1247.2
              c_only 0.812809            1906.8
          c_temporal 0.901778            1247.2
c_temporal_geometric 0.911177             555.0
                full 0.911177             555.0
```

Best compromise by F1 then false_new: `c_temporal_geometric`.
Geometric contribution diagnostic: full - c_temporal: F1 +0.009400, false_new -692.2.
Safe claim: feature channels are ablated in the controlled setting; geometry can be described as a safety channel only if its delta is small.
Forbidden claim: do not infer real-world inter-agent geometry from VisDrone single-camera results.
