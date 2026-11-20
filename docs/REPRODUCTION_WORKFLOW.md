# Reproduction workflow



**Associated article:** *From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging*  
**Journal:** *Applied Mathematics and Computation*  
**Article DOI:** https://doi.org/10.1016/j.amc.2026.130419  
**Research code package:** MATLAB Global-RBF and Local-RBF workflows  
**Repository:** https://github.com/HoaThiPham/RBF-Accuracy-to-Uncertainty  
**Archived research code (v1.0.1):** https://doi.org/10.5281/zenodo.24376136



This document separates the expensive model-generation stages from the manuscript-output stages. It preserves the execution sequence and output conventions of the research package. The public source archive does **not** include the restricted original geodetic observations or the prepared manuscript input MAT files. Consequently, access to the source code alone does not permit exact numerical reproduction of the manuscript's empirical results.

## Scope and requirements

* Run MATLAB from the **repository root**. The `setup\\\\\\\_project.m` helper configures the MATLAB search path; it does not change numerical settings or input/output files.
* The code requires MATLAB and toolboxes used by the workflows, including **Global Optimization Toolbox** (`ga`), **Parallel Computing Toolbox** (`parfor`, `parpool`, `gcp`), and **Statistics and Machine Learning Toolbox** (for example, `pdist2`, `cvpartition`). Record the exact tested MATLAB release, operating system, and installed toolbox versions before the public release.
* Inspect the **USER INPUT** settings in each selected runner before execution. Preserve the manuscript-matched configuration and random seeds when reproducing the published analysis.
* Computationally intensive production and bootstrap stages must run before dependent calibration and figure/table scripts. Do not assume an output file exists merely because its expected pathname is documented below.
* The prepared inputs and saved result files are distinct from the public code archive unless the release explicitly includes a verified, redistribution-authorised example.



## A. Working directory and prepared inputs

Run all workflows with MATLAB set to the repository/project root:

```matlab
projectRoot = setup\\\\\\\_project();
cd(projectRoot);
```

Required prepared inputs:

```text
OutPutSplitTrainTest/InputData\\\\\\\_Case1.mat
OutPutSplitTrainTest/InputData\\\\\\\_Case2.mat
```

containing MATLAB variables:

```text
InputData\\\\\\\_Case1
InputData\\\\\\\_Case2
```

These are **required prepared inputs**, not files supplied in the public code ZIP. The structures must contain the training/test coordinates and GGM-dependent fields expected by the runners. This document does not claim that restricted original data can be reconstructed from the public archive. Researchers with authorised access must prepare compatible inputs according to the applicable data agreements and the actual runner expectations. Do not replace the empirical inputs with synthetic data when claiming reproduction of the manuscript's reported numerical results.

For data-access conditions and the responsible authorisation process, consult the **Data availability** statement in the associated manuscript. Requests for access do not guarantee approval.

## B. Global-RBF

### B1. Main production

Run:

```text
Run\\\\\\\_Global\\\\\\\_RBF/Run\\\\\\\_Global\\\\\\\_RBF\\\\\\\_Full\\\\\\\_Production\\\\\\\_Case1.m
Run\\\\\\\_Global\\\\\\\_RBF/Run\\\\\\\_Global\\\\\\\_RBF\\\\\\\_Full\\\\\\\_Production\\\\\\\_Case2.m
```

Current production outputs:

```text
OutPut\\\\\\\_RBF\\\\\\\_2026/GlobalRBF\\\\\\\_Case1\\\\\\\_Full\\\\\\\_TuneReg\\\\\\\_4.mat
OutPut\\\\\\\_RBF\\\\\\\_2026/GlobalRBF\\\\\\\_Case2\\\\\\\_Full\\\\\\\_TuneReg\\\\\\\_4.mat
```

### B2. Ordinary/non-spatial bootstrap

Run:

```text
Run\\\\\\\_Global\\\\\\\_RBF/Run\\\\\\\_GlobalRBF\\\\\\\_NonSpatialBootstrap.m
```

Outputs:

```text
OutPut\\\\\\\_RBF\\\\\\\_2026/Case1\\\\\\\_NonSpatialBootstrap.mat
OutPut\\\\\\\_RBF\\\\\\\_2026/Case2\\\\\\\_NonSpatialBootstrap.mat
```

This is the baseline uncertainty-resampling procedure.

### B3. Baseline variance calibration

Run:

```text
Run\\\\\\\_Global\\\\\\\_RBF/RunGlobalCalibrationForNonSpatialBoostrapVariance.m
```

It uses the production outputs from B1 and the non-spatial bootstrap outputs from B2.

### B4. Spatial-bootstrap sensitivity

When reproducing the spatial-resampling sensitivity analysis, run:

```text
Run\\\\\\\_Global\\\\\\\_RBF/Run\\\\\\\_GlobalRBF\\\\\\\_SpatialBootstrap.m
Run\\\\\\\_Global\\\\\\\_RBF/RunGlobalCalibrationForSpatialBoostrapVariance.m
```

This is a sensitivity analysis rather than the baseline uncertainty procedure.

## C. Local-RBF

### C1. Main production

Run:

```text
Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Main\\\\\\\_Local/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Full\\\\\\\_Production\\\\\\\_Case1.m
Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Main\\\\\\\_Local/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Full\\\\\\\_Production\\\\\\\_Case2.m
```

Canonical final outputs:

```text
OutPut\\\\\\\_RBF\\\\\\\_2026\\\\\\\_2/LocalRBF\\\\\\\_Case1\\\\\\\_Full\\\\\\\_TuneReg.mat
OutPut\\\\\\\_RBF\\\\\\\_2026\\\\\\\_2/LocalRBF\\\\\\\_Case2\\\\\\\_Full\\\\\\\_TuneReg.mat
```

The current production radius grid is 50–200 km in 10-km increments for each kernel. Configuration identities are retained for intermediate Stage-1/Stage-2 checkpointing; final production files use the canonical names above.

### C2. Local bootstrap

Run:

```text
Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Main\\\\\\\_Local/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Bootstrap.m
```

The current default configuration uses 200 bootstrap replicates, `TrainingMode = 'pseudo-unseen'`, zero spatial segments (ordinary local bootstrap), and separate configuration output directories.

With the current defaults, the bootstrap results are written under:

```text
OutPut\\\\\\\_RBF\\\\\\\_2026\\\\\\\_2/LocalBootstrap\\\\\\\_pseudo\\\\\\\_unseen\\\\\\\_B200\\\\\\\_S0/
```

with:

```text
Case1\\\\\\\_Local\\\\\\\_Bootstrap.mat
Case2\\\\\\\_Local\\\\\\\_Bootstrap.mat
```

### C3. Local variance calibration

Run:

```text
Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Main\\\\\\\_Local/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Variance\\\\\\\_Calibration.m
```

With the current default configuration, this reads:

```text
OutPut\\\\\\\_RBF\\\\\\\_2026\\\\\\\_2/LocalBootstrap\\\\\\\_pseudo\\\\\\\_unseen\\\\\\\_B200\\\\\\\_S0/
```

and writes calibrated outputs under:

```text
OutPut\\\\\\\_RBF\\\\\\\_2026\\\\\\\_2/LocalVarianceCalibration\\\\\\\_pseudo\\\\\\\_unseen\\\\\\\_B200\\\\\\\_S0/
```

## D. Transfer sensitivity

Alternative transfer workflows are under:

```text
Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Transfer\\\\\\\_Sensitivity/
```

They provide separate IDW and nearest-neighbour production, bootstrap, and calibration runs.

## E. Ablation / robustness

Use:

```text
Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Ablation\\\\\\\_Sensitivity/
```

The master experiment is:

```text
Run\\\\\\\_Ablation\\\\\\\_RandomRobustness\\\\\\\_M0\\\\\\\_MP\\\\\\\_MR\\\\\\\_vs\\\\\\\_MPR.m
```

Definitions:

```text
M0  : fixed neighbourhood + fixed parameters
MP  : fixed neighbourhood + adaptive parameters
MR  : adaptive neighbourhood + fixed parameters
MPR : adaptive neighbourhood + adaptive parameters
```

`MPR` is the complete Local-RBF framework and reuses the completed main Local-RBF results.

## F. Manuscript outputs

After the required saved results exist, scripts/functions under:

```text
Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/
```

generate the manuscript figures and tables.

Supporting collectors are under:

```text
Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Support\\\\\\\_Functions\\\\\\\_For\\\\\\\_Figures\\\\\\\_And\\\\\\\_Tables/
```

Most exporters read saved results rather than rerunning the expensive RBF calculations.

The final Global spatial-bootstrap Table C.4 uses:

```text
6 current GGMs x 5 kernels = 30 combinations per case
```

and excludes legacy `sgg1`.

### Variance-figure input and output

`Figure\\\\\\\_RBF\\\\\\\_Variance\\\\\\\_FigC10\\\\\\\_C17.m` reads its numerical input from:

```text
OutPut\\\\\\\_RBF\\\\\\\_2026\\\\\\\_2/SpatialVariance/
```

and writes the exported figures to:

```text
OutPut\\\\\\\_RBF\\\\\\\_2026\\\\\\\_2/Manuscript/figures/
```

## G. Public-release portability

The current workflows are internally consistent with the documented project-root execution convention. Some scripts use Windows-style backslashes in relative path literals. Therefore **cross-platform execution has not been established by this documentation**. Before a cross-platform archival release, remaining Windows-style relative path literals may be converted to `fullfile(...)`, followed by testing to confirm that outputs are unchanged. Do not change scientific calculations solely for repository packaging.

## H. What can and cannot be reproduced from the public repository

|Task|Public source code alone|Additional requirements|
|-|-|-|
|Inspect Global-/Local-RBF implementation and configuration|Yes|MATLAB source and documentation|
|Execute main production experiments and reproduce reported numerical results|No|Authorised prepared `InputData\\\\\\\_Case1.mat` and `InputData\\\\\\\_Case2.mat`, matching environment and configurations|
|Execute bootstrap, variance calibration, transfer sensitivity, and ablation analyses|No|Relevant prepared inputs and/or prerequisite production outputs, as expected by the scripts|
|Regenerate principal figures and tables|Not necessarily|Saved numerical results in the script-specific expected paths; some exporters require additional derived inputs|
|Verify correctness using synthetic examples|Not guaranteed by this release|A separately verified synthetic example/test suite would be needed; none is asserted here|

The repository documents **methodological availability**, not unrestricted access to the input data or automatic reproduction of empirical findings. A figure exported successfully from saved data is not independent validation of the earlier computational stages.


## I. Release verification

1. Confirm that the archived source files and script paths match the code used to produce the published results.
2. Document the MATLAB, toolbox, and operating-system versions actually used, and verify path portability on the intended supported platforms.
3. Check released files and Git history for restricted observations, coordinate metadata, intermediate MAT files, confidential paths, and unauthorised outputs.
4. Verify that the documentation accurately describes required inputs and dependencies without implying that restricted data are publicly available.
5. Cross-check the article's figures and tables against the corresponding generated outputs, documenting any discrepancies.
6. Verify that the archived release matches the version-specific Zenodo record cited in the article.

## J. Associated publication and citation

**Article:** *From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging*  
**Authors:** Hoa Thi Pham; Sten Claessens; Ngoc-Tu Nguyen; Yangkang Yu; Sonny Pham; Martin Van Kranendonk  
**Journal:** *Applied Mathematics and Computation*  
**Article DOI:** https://doi.org/10.1016/j.amc.2026.130419  
**Corresponding author:** Hoa Thi Pham (`thihoa.pham@curtin.edu.au`; `pthoa.tdbd@hunre.edu.vn`)  
**GitHub repository:** https://github.com/HoaThiPham/RBF-Accuracy-to-Uncertainty  
**Archived research code (v1.0.1):** https://doi.org/10.5281/zenodo.24376136

When using or adapting this research code, cite both the published article and the version-specific Zenodo archive corresponding to the code used.

**Licence:** Non-Commercial Academic Research Code License — Version 1.0.  
**Copyright:** © 2026 Hoa Thi Pham.

The research code and restricted geodetic data are subject to separate access and reuse conditions.
