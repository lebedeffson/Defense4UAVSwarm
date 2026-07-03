# Defense4UAVSwarm v2 Smoke Bundle

## Commit
feature/v2-swarm-tnorm-xai, v2.3 EigenCAM smoke.

## Dataset Type
Pseudo-swarm over VisDrone VID.

## Agents
3 synthetic agents, synchronized by same frame id, projected by known affine transforms to reference agent.

## Features
c_i: detector confidence / pseudo confidence.
k_i: neutral in current smoke.
s_i: inter-agent IoU in reference frame.
x_i: selective EigenCAM-like saliency energy inside bbox.

## T-norms
min, product, Lukasiewicz.

## XAI
Method: eigencam fallback smoke.

## Current Status
Pseudo-swarm, s_i, S2 no-XAI smoke, and S3 XAI smoke are implemented and checked.

## Limitations
Current v2 is smoke/proof-of-concept only. No full calibration or holdout yet.
