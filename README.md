# From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging

**Official MATLAB implementation accompanying the research manuscript**

This repository contains the research code associated with the manuscript **“From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging”**, submitted to *Applied Mathematics and Computation*. It includes the core Global-RBF and Local-RBF routines, production runners, bootstrap uncertainty analysis, variance calibration, transfer-sensitivity and ablation experiments, and scripts used to generate manuscript figures and tables.

## Associated publication

**Title:** From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging  
**Journal:** *Applied Mathematics and Computation* (Elsevier)  
**Publication status:** \[Confirm current editorial status]  
**Article DOI:** \[Add when assigned]

**Authors (manuscript order):**

1. Hoa Thi Pham¹˒² (corresponding author)
2. Sten Claessens²
3. Ngoc-Tu Nguyen³˒⁴
4. Yangkang Yu²
5. Sonny Pham⁵
6. Martin Van Kranendonk²

**Affiliations:**

1. Faculty of Surveying, Mapping and Geographic Information, Hanoi University of Natural Resources and Environment, Hanoi, Vietnam.
2. School of Earth and Planetary Sciences, Curtin University, Perth, Australia.
3. University of Information Technology, Ho Chi Minh, Vietnam.
4. Vietnam National University Ho Chi Minh City, Ho Chi Minh, Vietnam.
5. School of Electrical Engineering, Computing and Mathematical Sciences, Curtin University, Perth, Australia.

**Corresponding author:** Hoa Thi Pham  
**Emails:** thihoa.pham@curtin.edu.au; pthoa.tdbd@hunre.edu.vn

**GitHub repository:** \[Add public repository URL]  
**Planned research code release:** `v1.0.0` (planned; confirm at release)  
**Zenodo version-specific DOI:** \[Add DOI after archive publication]

**Relationship to the article:** This package provides the research code accompanying the above manuscript, not a separate manuscript. The version-specific Zenodo archive should identify the exact release associated with the published numerical results.

## 1\. Implemented analyses

The code supports two RBF regimes:

* **Global RBF:** one RBF model is fitted over the complete training set for each GGM–kernel combination.
* **Local RBF:** point-specific neighbourhoods and RBF parameters are selected locally, followed by transfer of trained settings to independent test locations.

The manuscript configuration uses **six GGMs**:

1. `EGM2008`
2. `ggmplus`
3. `eigen6c4`
4. `geco`
5. `sgg2`
6. `xgm19`

and **five RBF kernels**:

1. Gaussian
2. Multiquadric
3. Thin-plate spline
4. Cubic
5. Linear

Thus, summaries over all current GGM–kernel combinations contain **30 combinations per case**. Legacy `sgg1` results are not part of the current manuscript configuration.

## 2\. Repository structure

```text
RBF\_Code\_Package/
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

The `+common`, `+globalrbf`, and `+localrbf` directories are MATLAB package folders. Their **parent directory** must be on the MATLAB path.

## 3\. MATLAB requirements

The code uses MATLAB plus at least:

* **Global Optimization Toolbox** — `ga`
* **Parallel Computing Toolbox** — `parfor`, `parpool`, `gcp`
* **Statistics and Machine Learning Toolbox** — e.g. `pdist2`, `cvpartition`, percentile utilities

Before public release, authors should complete the exact environment used for the manuscript computations:

```text
MATLAB release: \[TO BE FILLED BY AUTHORS]
Operating system: \[TO BE FILLED BY AUTHORS]
```

## 4\. Setup and execution convention

Run the manuscript scripts with the MATLAB current folder set to the repository root.

Recommended startup:

```matlab
projectRoot = setup\_project();
cd(projectRoot);
```

`setup\_project.m` adds the package, runner, and manuscript-output folders to the MATLAB search path. It does not change scientific parameters, input files, output files, or the current folder.

The current manuscript code is internally consistent with this project-root execution convention. Some scripts retain Windows-style backslashes in relative path literals; these can be converted to `fullfile(...)` later for a platform-independent public release without changing the scientific calculations.

## 5\. Required prepared inputs

The principal production scripts expect:

```text
OutPutSplitTrainTest/
├── InputData\_Case1.mat   -> variable: InputData\_Case1
└── InputData\_Case2.mat   -> variable: InputData\_Case2
```

The code expects training/test coordinates and GGM-dependent target fields inside these structures.

The original geodetic datasets are subject to access and redistribution restrictions and are not included in this repository. Access to the corresponding prepared inputs requires appropriate authorisation. The absence of these inputs limits independent reproduction of the manuscript numerical results; the public source code alone does not recreate restricted observational data.

## 6\. Reproduction order

### 6.1 Global-RBF

Run:

1. `Run\_Global\_RBF/Run\_Global\_RBF\_Full\_Production\_Case1.m`
2. `Run\_Global\_RBF/Run\_Global\_RBF\_Full\_Production\_Case2.m`
3. `Run\_Global\_RBF/Run\_GlobalRBF\_NonSpatialBootstrap.m`
4. `Run\_Global\_RBF/RunGlobalCalibrationForNonSpatialBoostrapVariance.m`
5. optionally, `Run\_Global\_RBF/Run\_GlobalRBF\_SpatialBootstrap.m`
6. optionally, `Run\_Global\_RBF/RunGlobalCalibrationForSpatialBoostrapVariance.m`

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

The spatial bootstrap is a **sensitivity analysis**; the ordinary/non-spatial bootstrap is retained as the main uncertainty analysis.

### 6.2 Local-RBF

Use:

```text
Run\_Local\_RBF/Run\_Main\_Local/
```

in this order:

1. `Run\_Local\_RBF\_Full\_Production\_Case1.m`
2. `Run\_Local\_RBF\_Full\_Production\_Case2.m`
3. `Run\_Local\_RBF\_Bootstrap.m`
4. `Run\_Local\_RBF\_Variance\_Calibration.m`

The canonical final Local-RBF results are:

```text
OutPut\_RBF\_2026\_2/LocalRBF\_Case1\_Full\_TuneReg.mat
OutPut\_RBF\_2026\_2/LocalRBF\_Case2\_Full\_TuneReg.mat
```

The main Local-RBF production grid uses candidate radii from **50 to 200 km in 10-km increments** for each kernel. Stage-1/Stage-2 checkpoints retain configuration identities, whereas the final production result files use the canonical names above.

### 6.3 Transfer sensitivity

Alternative transfer experiments are under:

```text
Run\_Local\_RBF/Run\_Check\_Transfer\_Sensitivity/
```

They evaluate:

* nearest-neighbour transfer;
* inverse-distance-weighted (IDW) transfer.

These are sensitivity checks and are separate from the main Local-RBF production workflow.

### 6.4 Ablation analysis

Ablation scripts are under:

```text
Run\_Local\_RBF/Run\_Check\_Ablation\_Sensitivity/
```

with:

* `M0`: fixed neighbourhood + fixed kernel parameters
* `MP`: fixed neighbourhood + adaptive point-specific parameters
* `MR`: adaptive neighbourhood + fixed kernel parameters
* `MPR`: adaptive neighbourhood + adaptive parameters (complete Local-RBF framework)

The master runner is:

```text
Run\_Ablation\_RandomRobustness\_M0\_MP\_MR\_vs\_MPR.m
```

`MPR` reuses the completed main Local-RBF results.

## 7\. Bootstrap and uncertainty calibration

Both Global- and Local-RBF workflows include bootstrap uncertainty estimation followed by variance calibration.

The final Global spatial-bootstrap sensitivity summary uses:

```text
6 GGMs x 5 kernels = 30 combinations per case
```

Legacy `sgg1` entries that may remain in older saved result files are excluded from the final manuscript Table C.4 analysis.

## 8\. Manuscript figures and tables

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

## 9\. Reproducibility safeguards

The implementation includes:

* fixed RNG seeds in relevant bootstrap/sensitivity workflows;
* saved bootstrap seed information;
* explicit six-GGM/five-kernel manuscript configuration;
* separation of model selection/training from independent test evaluation;
* checkpointed Local-RBF calculations;
* saved provenance and numerical-condition diagnostics;
* explicit ablation definitions;
* manuscript exporters based on saved numerical results.

## 10\. Before first use

1. Extract the repository.
2. Start MATLAB.
3. Run `projectRoot = setup\_project();`.
4. Run `cd(projectRoot);`.
5. Place or construct `InputData\_Case1.mat` and `InputData\_Case2.mat` under `OutPutSplitTrainTest/`.
6. Review the **USER INPUT** sections of the selected runner.
7. Run the production calculations before bootstrap/calibration.
8. Generate manuscript figures/tables after their required saved results exist.

## 11\. Public research code release and versioning

This repository provides the MATLAB research code and documentation accompanying the scientific article **“From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging.”** The package supports scientific transparency, reproducibility, independent verification, and further non-commercial academic research.

The version associated with the manuscript will be tagged on GitHub (e.g., `v1.0.0`) and archived on Zenodo with a **version-specific DOI**. This archived snapshot identifies the research code package associated with the reported numerical results; later changes to the GitHub default branch do not modify the archived version.

**Before making the repository public:**

* Confirm the MATLAB release, operating system, and required toolboxes.
* Audit the package and Git history for restricted geodetic data, sensitive metadata, or confidential intermediate results.
* Include the final, approved **Non-Commercial Academic Research Code License**; verify rights to distribute all included components.
* Verify code creator and contributor metadata in `CITATION.cff` (copyright ownership and code authorship are distinct).
* Check file paths and the documented execution order on the supported operating system.
* Verify that the release tag points to the implementation used for the manuscript results.

Subsequent documentation or code improvements can be made on GitHub. To archive changed files on Zenodo, create a new versioned release; the original version-specific DOI remains associated with the original snapshot.

## 12\. Data and code availability

### Data availability

The original geodetic data analysed in this study are subject to access and redistribution restrictions. Requests for access must follow the relevant authorisation procedures and approval requirements. These datasets are **not included** in the GitHub repository or the associated Zenodo archive.

### Code availability

The MATLAB research code package is intended to be made publicly accessible through GitHub, with the release corresponding to the published results archived on Zenodo.

* **GitHub:** \[Add the official public repository URL after release]
* **Zenodo:** \[Add the version-specific DOI after archive publication]

The package includes Global-RBF and Local-RBF workflows, parameter optimisation, bootstrap uncertainty calibration, sensitivity analyses, configuration settings, random seeds, and scripts for generating the principal figures and tables.

The code is intended to be distributed under the **Non-Commercial Academic Research Code License**, following approval of its final terms. The licence is designed to permit non-commercial academic use, modification, development of new research methods, sharing of permitted derivatives, and publication of derived scientific findings subject to attribution and the applicable licence conditions. Commercial exploitation requires separate written permission from the copyright holder.

The original restricted geodetic data are not redistributed. Access to the code does not, by itself, enable exact reproduction of the study's restricted-data numerical results.

## 13\. How to cite

Please cite the associated research article and, when using or extending the MATLAB research code, the corresponding **version-specific Zenodo record**. Follow the attribution requirements of the final research code licence.

### Associated research article

Pham, H. T., Claessens, S., Nguyen, N.-T., Yu, Y., Pham, S., and Van Kranendonk, M. *From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging.* **Applied Mathematics and Computation**. \[Publication year, volume, article number, and DOI to be added.]

### Archived MATLAB research code package

Pham, H. T. \[Code creator attribution to be confirmed]. *From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging — MATLAB Research Code and Reproducibility Package.* Version `v1.0.0` (planned). Zenodo. \[Version-specific DOI to be added.]

The code creator list in the Zenodo record and `CITATION.cff` must reflect actual contributions; it need not be identical to the manuscript author list or the copyright holder.

### Copyright and research use

**Copyright © 2026 Hoa Thi Pham.** All rights reserved except as expressly licensed under the **Non-Commercial Academic Research Code License**, once finalised and approved.

The research code may be used, modified, and extended for permitted non-commercial academic research, including development and publication of new methods, subject to the licence terms and attribution requirements. Commercial exploitation requires prior written permission from the copyright holder.

**Copyright holder:** Hoa Thi Pham  
**Contact:** thihoa.pham@curtin.edu.au; pthoa.tdbd@hunre.edu.vn

