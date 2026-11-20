From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging 
MATLAB Research Code and Reproducibility Package
This repository provides the MATLAB research code accompanying the article to be published in Applied Mathematics and Computation (Elsevier).
The package includes Global-RBF and Local-RBF implementations, parameter optimisation, bootstrap uncertainty calibration, transfer-sensitivity and ablation analyses, and scripts for generating the principal manuscript figures and tables.
Associated Publication
Journal: Applied Mathematics and Computation (Elsevier)
Article DOI: To be added when assigned.
Authors
Hoa Thi Pham¹˒² (corresponding author)
Sten Claessens²
Ngoc-Tu Nguyen³˒⁴
Yangkang Yu²
Sonny Pham⁵
Martin Van Kranendonk²
Affiliations
¹ Faculty of Surveying, Mapping and Geographic Information, Hanoi University of Natural Resources and Environment, Hanoi, Vietnam.
² School of Earth and Planetary Sciences, Curtin University, Perth, Australia.
³ University of Information Technology, Ho Chi Minh, Vietnam.
⁴ Vietnam National University Ho Chi Minh City, Ho Chi Minh, Vietnam.
⁵ School of Electrical Engineering, Computing and Mathematical Sciences, Curtin University, Perth, Australia.
Corresponding author: Hoa Thi Pham
Email: thihoa.pham@curtin.edu.au; pthoa.tdbd@hunre.edu.vn
1. Implemented analyses
The code supports two RBF regimes:
Global RBF: one RBF model is fitted over the complete training set for each GGM–kernel combination.
Local RBF: point-specific neighbourhoods and RBF parameters are selected locally, followed by transfer of trained settings to independent test locations.
The manuscript configuration uses six GGMs:
`EGM2008`
`ggmplus`
`eigen6c4`
`geco`
`sgg2`
`xgm19`
and five RBF kernels:
Gaussian
Multiquadric
Thin-plate spline
Cubic
Linear
Thus, summaries over all current GGM–kernel combinations contain 30 combinations per case. Legacy `sgg1` results are not part of the current manuscript configuration.
2. Repository structure
```text
RBF-Accuracy-to-Uncertainty/
├── README.md
├── CITATION.cff                   Citation metadata
├── LICENSE                        Research code usage terms
├── setup\_project.m
├── +common/                         Shared numerical RBF/calibration utilities
├── +globalrbf/                      Global-RBF implementation
├── +localrbf/                       Local-RBF implementation
├── Run\_Global\_RBF/                  Global-RBF production/bootstrap/calibration
├── Run\_Local\_RBF/
│   ├── Run\_Main\_Local/              Main Local-RBF workflow
│   ├── Run\_Check\_Ablation\_Sensitivity/
│   │                                 M0, MP, MR, and MPR experiments
│   └── Run\_Check\_Transfer\_Sensitivity/
│                                     IDW and nearest-neighbour sensitivity
├── Fig\_And\_Table\_RBF/               Manuscript figure/table exporters
│   └── Support\_Functions\_For\_Figures\_And\_Tables/
│                                     Supporting collectors
└── docs/
    ├── REPRODUCTION\_WORKFLOW.md
    └── REPOSITORY\_MAP.md
```
The `+common`, `+globalrbf`, and `+localrbf` directories are MATLAB package folders. Their parent directory must be on the MATLAB path.
3. MATLAB requirements
The code uses MATLAB plus at least:
Global Optimization Toolbox — `ga`
Parallel Computing Toolbox — `parfor`, `parpool`, `gcp`
Statistics and Machine Learning Toolbox — e.g. `pdist2`, `cvpartition`, percentile utilities
Before claiming the package has been tested in a specific environment, complete the following details:
```text
MATLAB release: \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\[TO BE FILLED BY AUTHORS]
Operating system: \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\[TO BE FILLED BY AUTHORS]
```
4. Setup and execution convention
Run the manuscript scripts with the MATLAB current folder set to the repository root.
Recommended startup:
```matlab
projectRoot = setup\_project();
cd(projectRoot);
```
`setup\_project.m` adds the package, runner, and manuscript-output folders to the MATLAB search path. It does not change scientific parameters, input files, output files, or the current folder.
The current manuscript code is internally consistent with this project-root execution convention. Some scripts retain Windows-style backslashes in relative path literals; these can be converted to `fullfile(...)` later for a platform-independent public release without changing the scientific calculations.
5. Required prepared inputs
The principal production scripts expect:
```text
OutPutSplitTrainTest/
├── InputData\_Case1.mat   -> variable: InputData\_Case1
└── InputData\_Case2.mat   -> variable: InputData\_Case2
```
The code expects training/test coordinates and GGM-dependent target fields inside these structures.
The original geodetic datasets are subject to access and redistribution restrictions and are not included in this repository. Access to the corresponding prepared inputs requires appropriate authorisation. The absence of these inputs limits independent reproduction of the manuscript numerical results; the public source code alone does not recreate restricted observational data.
6. Reproduction order
6.1 Global-RBF
Run:
`Run\_Global\_RBF/Run\_Global\_RBF\_Full\_Production\_Case1.m`
`Run\_Global\_RBF/Run\_Global\_RBF\_Full\_Production\_Case2.m`
`Run\_Global\_RBF/Run\_GlobalRBF\_NonSpatialBootstrap.m`
`Run\_Global\_RBF/RunGlobalCalibrationForNonSpatialBoostrapVariance.m`
optionally, `Run\_Global\_RBF/Run\_GlobalRBF\_SpatialBootstrap.m`
optionally, `Run\_Global\_RBF/RunGlobalCalibrationForSpatialBoostrapVariance.m`
The main Global-RBF production files are:
```text
OutPut\_RBF\_2026/GlobalRBF\_Case1\_Full\_TuneReg\_4.mat
OutPut\_RBF\_2026/GlobalRBF\_Case2\_Full\_TuneReg\_4.mat
```
The ordinary/non-spatial bootstrap files are:
```text
OutPut\_RBF\_2026/Case1\_NonSpatialBootstrap.mat
OutPut\_RBF\_2026/Case2\_NonSpatialBootstrap.mat
```
The spatial bootstrap is a sensitivity analysis; the ordinary/non-spatial bootstrap is retained as the main uncertainty analysis.
6.2 Local-RBF
Use:
```text
Run\_Local\_RBF/Run\_Main\_Local/
```
in this order:
`Run\_Local\_RBF\_Full\_Production\_Case1.m`
`Run\_Local\_RBF\_Full\_Production\_Case2.m`
`Run\_Local\_RBF\_Bootstrap.m`
`Run\_Local\_RBF\_Variance\_Calibration.m`
The canonical final Local-RBF results are:
```text
OutPut\_RBF\_2026\_2/LocalRBF\_Case1\_Full\_TuneReg.mat
OutPut\_RBF\_2026\_2/LocalRBF\_Case2\_Full\_TuneReg.mat
```
The main Local-RBF production grid uses candidate radii from 50 to 200 km in 10-km increments for each kernel. Stage-1/Stage-2 checkpoints retain configuration identities, whereas the final production result files use the canonical names above.
6.3 Transfer sensitivity
Alternative transfer experiments are under:
```text
Run\_Local\_RBF/Run\_Check\_Transfer\_Sensitivity/
```
They evaluate:
nearest-neighbour transfer;
inverse-distance-weighted (IDW) transfer.
These are sensitivity checks and are separate from the main Local-RBF production workflow.
6.4 Ablation analysis
Ablation scripts are under:
```text
Run\_Local\_RBF/Run\_Check\_Ablation\_Sensitivity/
```
with:
`M0`: fixed neighbourhood + fixed kernel parameters
`MP`: fixed neighbourhood + adaptive point-specific parameters
`MR`: adaptive neighbourhood + fixed kernel parameters
`MPR`: adaptive neighbourhood + adaptive parameters (complete Local-RBF framework)
The master runner is:
```text
Run\_Ablation\_RandomRobustness\_M0\_MP\_MR\_vs\_MPR.m
```
`MPR` reuses the completed main Local-RBF results.
7. Bootstrap and uncertainty calibration
Both Global- and Local-RBF workflows include bootstrap uncertainty estimation followed by variance calibration.
The final Global spatial-bootstrap sensitivity summary uses:
```text
6 GGMs x 5 kernels = 30 combinations per case
```
Legacy `sgg1` entries that may remain in older saved result files are excluded from the final manuscript Table C.4 analysis.
8. Manuscript figures and tables
Scripts under:
```text
Fig\_And\_Table\_RBF/
```
generate manuscript figures and tables from saved numerical results. Supporting collectors are under:
```text
Fig\_And\_Table\_RBF/Support\_Functions\_For\_Figures\_And\_Tables/
```
These exporters generally reuse saved production results rather than rerunning expensive RBF optimization or bootstrap calculations.
The current figure scripts embed `compact\_subplot` locally where required; there is no external `tight\_subplot.m` dependency.
`Figure\_RBF\_Variance\_FigC10\_C17.m` reads its numerical input from `OutPut\_RBF\_2026\_2/SpatialVariance/` and writes the exported figures to the canonical manuscript-output directory `OutPut\_RBF\_2026\_2/Manuscript/figures/`.
9. Reproducibility safeguards
The implementation includes:
fixed RNG seeds in relevant bootstrap/sensitivity workflows;
saved bootstrap seed information;
explicit six-GGM/five-kernel manuscript configuration;
separation of model selection/training from independent test evaluation;
checkpointed Local-RBF calculations;
saved provenance and numerical-condition diagnostics;
explicit ablation definitions;
manuscript exporters based on saved numerical results.
10. Before first use
Extract the repository.
Start MATLAB.
Run `projectRoot = setup\_project();`.
Run `cd(projectRoot);`.
Place or construct `InputData\_Case1.mat` and `InputData\_Case2.mat` under `OutPutSplitTrainTest/`.
Review the USER INPUT sections of the selected runner.
Run the production calculations before bootstrap/calibration.
Generate manuscript figures/tables after their required saved results exist.
11. Research code releases and versioning
This repository provides the MATLAB research code and documentation accompanying the article From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging.
Research code releases are versioned on GitHub and archived on Zenodo to support reproducibility, transparency, and long-term accessibility.
Each archived release is assigned a version-specific DOI, allowing researchers to identify and cite the exact code version used in their work.
Updates to the GitHub repository do not modify previously archived Zenodo releases. Changes to the archived research code are documented through subsequent versioned releases.
12. Data and code availability
Data availability
The geodetic data analysed in this study are subject to access and redistribution restrictions imposed by the relevant Vietnamese data providers and cannot be made publicly available. Requests for access may be directed to the corresponding author and will be considered subject to approval by the relevant data custodians and applicable authorisation requirements.
The original geodetic observations and restricted prepared case inputs are not included in this GitHub repository or its associated Zenodo archives.
Code availability
GitHub repository: https://github.com/HoaThiPham/RBF-Accuracy-to-Uncertainty
The repository provides the MATLAB research code for Global-RBF and Local-RBF interpolation, including parameter optimisation, bootstrap uncertainty calibration, sensitivity analyses, configuration settings, random seeds, and manuscript figure and table generation.
The research code is distributed under the accompanying Non-Commercial Academic Research Code License.
The original geodetic data are subject to access restrictions and are not included. Reproducing the reported numerical results requires authorised access to the corresponding datasets and a compatible computational environment.
13. How to cite
If you use or extend this research code, please cite the associated research article and the version-specific Zenodo DOI of the release used
Associated research article
Pham, H. T., Claessens, S., Nguyen, N.-T., Yu, Y., Pham, S., and Van Kranendonk, M. From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging. Applied Mathematics and Computation. [Publication year, volume, article number, and DOI to be added.]
Archived MATLAB research code package
From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging — MATLAB Research Code and Reproducibility Package.
The citation metadata and version-specific DOI are provided on the corresponding Zenodo release record
Copyright and research use
Copyright © 2026 Hoa Thi Pham.
All rights reserved except as expressly licensed in the accompanying `LICENSE` file.
Copyright holder: Hoa Thi Pham
Contact: thihoa.pham@curtin.edu.au; pthoa.tdbd@hunre.edu.vn
