# Release compatibility validation

The release was compared with the original implementation, using the final SubsetA checkpoint at epoch 12 / iteration 33516.

- Original checkpoint SHA256: `da89433c07d3e1019d260b554b6febb577fc15b0ce805ebb0350d2d3de98d841`.
- All 1,023 parameters/buffers survive the public naming conversion with tensor-exact equality and strict loading.
- Exported checkpoint loads with `weights_only=True`; optimizer and private metadata are absent.
- CPU first-frame and history-frame fixtures compare QWM, BWM, BEV fusion, all six decoder layers, relationship prediction, raw ordered adjacency caching and prediction formatting with zero numerical tolerance.
- The complete `forward_test` entry (including image backbone/FPN, BEV encoder, ego-motion handling and result decoding) also produces tensor-exact predictions on first/history-frame CPU fixtures. A new-scene fixture verifies query/topology history reset.
- Fixtures use an 8×16 BEV grid to exercise the state paths economically. The published model/configuration retains the accepted 100×200 grid.
- Python source passes compilation and undefined-name checks; the CLI imports without loading training frameworks.

The CPU fixtures verify compatibility of the extraction. A new full 4,806-frame GPU run was not performed for this code release. This document does not claim that CPU fixtures replace dataset-level evaluation or measure release GPU throughput.
