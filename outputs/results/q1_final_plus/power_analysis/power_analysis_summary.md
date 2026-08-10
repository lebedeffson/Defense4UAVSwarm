# Q1 Power Analysis Summary

Comparison: ByteTrack vs ByteTrack + geometry trust.
Metric: `false_new_tracks`.

Sequence-level deltas are method_b - method_a:

```text
       sequence_id  false_new_tracks_a  false_new_tracks_b  delta
uav0000086_00000_v                1116                 848 -268.0
uav0000117_02622_v                1186                 858 -328.0
uav0000137_00458_v                1296                 903 -393.0
uav0000182_00000_v                1558                1054 -504.0
uav0000268_05773_v                 397                 277 -120.0
uav0000305_00000_v                 345                 218 -127.0
uav0000339_00001_v                 940                 641 -299.0
```

mean_delta_false_new: -291.285714
std_delta_false_new: 137.762736
paired_effect_size: -2.114401
estimated_required_sequences: 2.0
bootstrap_n_ci95: [1.0, 4.0]
estimate_stable: no

Required answers:
1. Can we robustly claim 15-20 sequences? No. The estimate should not be presented as a stable 15-20 sequence requirement.
2. Calculation: normal approximation for paired mean delta using observed paired effect size.
3. Stability: unstable if based on only 7 sequences or wide bootstrap CI.
4. Safe article sentence: Power estimate is unstable because only 7 independent VisDrone sequences are available; report it as an exploratory calculation, not as a firm sample-size requirement.

Bootstrap finite samples: 1000
