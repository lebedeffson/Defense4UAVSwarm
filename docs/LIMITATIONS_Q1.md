# Q1 Closure Limitations

The Q1 criticism is only fully closed when U2UData/U2UData+ or another genuine multi-agent UAV/cooperative dataset is available locally and the v7 experiment produces real metrics.

Do not claim real swarm validation from:

* VisDrone pseudo-swarm only;
* proxy manifest replay only;
* blocked U2UData reports;
* AirSim/custom simulation without clearly calling it simulation.

Current scripts enforce this by writing `blocked_*` statuses when dataset, manifest, or detections are missing.

Acceptable claims:

* controlled pseudo-swarm stress test;
* temporal trust replay on existing artifacts;
* U2UData integration tooling;
* real U2UData validation only after metrics are generated from a local U2UData manifest.
