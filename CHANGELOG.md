# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [ApDepth]
### Added
- A feature refinement module has been added within the UNet architecture of Stable Diffusion v2. This module refines low-frequency features and enhances high-frequency features to achieve improved edge sharpness.
### Result
- Available later

---

## [Marigold-depth-v1-0] - 2023-12-04
The Result is showed in table
| Method | # Training Samples | | NYUv2 | | KITTI | | ETH3D | | ScanNet | | DIODE | | Avg. Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Real** | **Synthetic** | **AbsRel↓** | **δ1↑** | **AbsRel↓** | **δ1↑** | **AbsRel↓** | **δ1↑** | **AbsRel↓** | **δ1↑** | **AbsRel↓** | **δ1↑** | |
| DiverseDepth [56] | 320K | — | 11.7 | 87.5 | 19.0 | 70.4 | 22.8 | 69.4 | 10.9 | 88.2 | 37.6 | 63.1 | 7.6 |
| MiDaS [35] | 2M | — | 11.1 | 88.5 | 23.6 | 63.0 | 18.4 | 75.2 | 12.1 | 84.6 | 33.2 | 71.5 | 7.3 |
| LeReS [57] | 300K | 54K | 9.0 | 91.6 | 14.9 | 78.4 | 17.1 | 77.7 | 9.1 | 91.7 | 27.1 | 76.6 | 5.2 |
| Omnidata [13] | 11.9M | 310K | 7.4 | 94.5 | 14.9 | 83.5 | 16.6 | 77.8 | 7.5 | 93.6 | 33.9 | 74.2 | 4.8 |
| HDN [60] | 300K | — | 6.9 | 94.8 | 11.5 | 86.7 | 12.1 | 83.3 | 8.0 | 93.9 | 24.6 | **78.0** | 3.2 |
| DPT [36] | 1.2M | 188K | 9.8 | 90.3 | 10.0 | 90.1 | 7.8 | 94.6 | 8.2 | 93.4 | **18.2** | 75.8 | 3.9 |
| Marigold (w/o) | —* | 74K | 6.0 | 95.9 | 10.5 | 90.4 | 7.1 | 95.1 | 6.9 | 94.5 | 31.0 | 77.2 | 2.5 |
| **Marigold** | | | **5.5** | **96.4** | **9.9** | **91.6** | **6.5** | **96.0** | **6.4** | **95.1** | 30.8 | 77.3 | **1.4** |
<hr>

![cover](doc/teaser_collage_transparant.png)