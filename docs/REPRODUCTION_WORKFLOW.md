Reproduction workflow
Associated manuscript: From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging  
Journal: Applied Mathematics and Computation  
Research code package: MATLAB Global-RBF and Local-RBF workflows  
Repository: https://github.com/HoaThiPham/RBF-Accuracy-to-Uncertainty  


Manuscript-matched release: `v1.0.1` (planned; confirm that this release has been published and verified before citing its DOI)
This document separates the expensive model-generation stages from the manuscript-output stages. It preserves the execution sequence and output conventions of the research package. The public source archive does not include the restricted original geodetic observations or the prepared manuscript input MAT files. Consequently, access to the source code alone does not permit exact numerical reproduction of the manuscript's empirical results.
Scope and requirements
Run MATLAB from the repository root. The `setup_project.m` helper configures the MATLAB search path; it does not change numerical settings or input/output files.
The code requires MATLAB and toolboxes used by the workflows, including Global Optimization Toolbox (`ga`), Parallel Computing Toolbox (`parfor`, `parpool`, `gcp`), and Statistics and Machine Learning Toolbox (for example, `pdist2`, `cvpartition`). Record the exact tested MATLAB release, operating system, and installed toolbox versions before the public release.
Inspect the USER INPUT settings in each selected runner before execution. Preserve the manuscript-matched configuration and random seeds when reproducing the published analysis.
Computationally intensive production and bootstrap stages must run before dependent calibration and figure/table scripts. Do not assume an output file exists merely because its expected pathname is documented below.
The prepared inputs and saved result files are distinct from the public code archive unless the release explicitly includes a verified, redistribution-authorised example.

A. Working directory and prepared inputs
Run all workflows with MATLAB set to the repository/project root:
```matlab
projectRoot = setup_project();
cd(projectRoot);
```
Required prepared inputs:
```text
OutPutSplitTrainTest/InputData_Case1.mat
OutPutSplitTrainTest/InputData_Case2.mat
```
containing MATLAB variables:
```text
InputData_Case1
InputData_Case2
```
These are required prepared inputs, not files supplied in the public code ZIP. The structures must contain the training/test coordinates and GGM-dependent fields expected by the runners. This document does not claim that restricted original data can be reconstructed from the public archive. Researchers with authorised access must prepare compatible inputs according to the applicable data agreements and the actual runner expectations. Do not replace the empirical inputs with synthetic data when claiming reproduction of the manuscript's reported numerical results.
For data-access conditions and the responsible authorisation process, consult the Data availability statement in the associated manuscript. Requests for access do not guarantee approval.
B. Global-RBF
B1. Main production
Run:
```text
Run_Global_RBF/Run_Global_RBF_Full_Production_Case1.m
Run_Global_RBF/Run_Global_RBF_Full_Production_Case2.m
```
Current production outputs:
```text
OutPut_RBF_2026/GlobalRBF_Case1_Full_TuneReg_4.mat
OutPut_RBF_2026/GlobalRBF_Case2_Full_TuneReg_4.mat
```
B2. Ordinary/non-spatial bootstrap
Run:
```text
Run_Global_RBF/Run_GlobalRBF_NonSpatialBootstrap.m
```
Outputs:
```text
OutPut_RBF_2026/Case1_NonSpatialBootstrap.mat
OutPut_RBF_2026/Case2_NonSpatialBootstrap.mat
```
This is the baseline uncertainty-resampling procedure.
B3. Baseline variance calibration
Run:
```text
Run_Global_RBF/RunGlobalCalibrationForNonSpatialBoostrapVariance.m
```
It uses the production outputs from B1 and the non-spatial bootstrap outputs from B2.
B4. Spatial-bootstrap sensitivity
When reproducing the spatial-resampling sensitivity analysis, run:
```text
Run_Global_RBF/Run_GlobalRBF_SpatialBootstrap.m
Run_Global_RBF/RunGlobalCalibrationForSpatialBoostrapVariance.m
```
This is a sensitivity analysis rather than the baseline uncertainty procedure.
C. Local-RBF
C1. Main production
Run:
```text
Run_Local_RBF/Run_Main_Local/Run_Local_RBF_Full_Production_Case1.m
Run_Local_RBF/Run_Main_Local/Run_Local_RBF_Full_Production_Case2.m
```
Canonical final outputs:
```text
OutPut_RBF_2026_2/LocalRBF_Case1_Full_TuneReg.mat
OutPut_RBF_2026_2/LocalRBF_Case2_Full_TuneReg.mat
```
The current production radius grid is 50–200 km in 10-km increments for each kernel. Configuration identities are retained for intermediate Stage-1/Stage-2 checkpointing; final production files use the canonical names above.
C2. Local bootstrap
Run:
```text
Run_Local_RBF/Run_Main_Local/Run_Local_RBF_Bootstrap.m
```
The current default configuration uses 200 bootstrap replicates, `TrainingMode = 'pseudo-unseen'`, zero spatial segments (ordinary local bootstrap), and separate configuration output directories.
With the current defaults, the bootstrap results are written under:
```text
OutPut_RBF_2026_2/LocalBootstrap_pseudo_unseen_B200_S0/
```
with:
```text
Case1_Local_Bootstrap.mat
Case2_Local_Bootstrap.mat
```
C3. Local variance calibration
Run:
```text
Run_Local_RBF/Run_Main_Local/Run_Local_RBF_Variance_Calibration.m
```
With the current default configuration, this reads:
```text
OutPut_RBF_2026_2/LocalBootstrap_pseudo_unseen_B200_S0/
```
and writes calibrated outputs under:
```text
OutPut_RBF_2026_2/LocalVarianceCalibration_pseudo_unseen_B200_S0/
```
D. Transfer sensitivity
Alternative transfer workflows are under:
```text
Run_Local_RBF/Run_Check_Transfer_Sensitivity/
```
They provide separate IDW and nearest-neighbour production, bootstrap, and calibration runs.
E. Ablation / robustness
Use:
```text
Run_Local_RBF/Run_Check_Ablation_Sensitivity/
```
The master experiment is:
```text
Run_Ablation_RandomRobustness_M0_MP_MR_vs_MPR.m
```
Definitions:
```text
M0  : fixed neighbourhood + fixed parameters
MP  : fixed neighbourhood + adaptive parameters
MR  : adaptive neighbourhood + fixed parameters
MPR : adaptive neighbourhood + adaptive parameters
```
`MPR` is the complete Local-RBF framework and reuses the completed main Local-RBF results.
F. Manuscript outputs
After the required saved results exist, scripts/functions under:
```text
Fig_And_Table_RBF/
```
generate the manuscript figures and tables.
Supporting collectors are under:
```text
Fig_And_Table_RBF/Support_Functions_For_Figures_And_Tables/
```
Most exporters read saved results rather than rerunning the expensive RBF calculations.
The final Global spatial-bootstrap Table C.4 uses:
```text
6 current GGMs x 5 kernels = 30 combinations per case
```
and excludes legacy `sgg1`.
Variance-figure input and output
`Figure_RBF_Variance_FigC10_C17.m` reads its numerical input from:
```text
OutPut_RBF_2026_2/SpatialVariance/
```
and writes the exported figures to:
```text
OutPut_RBF_2026_2/Manuscript/figures/
```
G. Public-release portability
The current workflows are internally consistent with the documented project-root execution convention. Some scripts use Windows-style backslashes in relative path literals. Therefore cross-platform execution has not been established by this documentation. Before a cross-platform archival release, remaining Windows-style relative path literals may be converted to `fullfile(...)`, followed by testing to confirm that outputs are unchanged. Do not change scientific calculations solely for repository packaging.
H. What can and cannot be reproduced from the public repository
Task	Public source code alone	Additional requirements
Inspect Global-/Local-RBF implementation and configuration	Yes	MATLAB source and documentation
Execute main production experiments and reproduce reported numerical results	No	Authorised prepared `InputData_Case1.mat` and `InputData_Case2.mat`, matching environment and configurations
Execute bootstrap, variance calibration, transfer sensitivity, and ablation analyses	No	Relevant prepared inputs and/or prerequisite production outputs, as expected by the scripts
Regenerate principal figures and tables	Not necessarily	Saved numerical results in the script-specific expected paths; some exporters require additional derived inputs
Verify correctness using synthetic examples	Not guaranteed by this release	A separately verified synthetic example/test suite would be needed; none is asserted here
The repository documents methodological availability, not unrestricted access to the input data or automatic reproduction of empirical findings. A figure exported successfully from saved data is not independent validation of the earlier computational stages.
I. Verification before the complete GitHub/Zenodo release
Confirm the released source files and script paths match the code used for the manuscript's final results.
Record MATLAB, toolbox, and operating-system versions actually used. Check path portability on the intended supported platform(s).
Review files and Git history for restricted raw/processed observations, coordinate metadata, intermediate MAT files, embedded confidential paths, and unauthorised outputs.
Verify that public documentation correctly identifies the required input structures and dependencies without promising access to restricted observations.
Verify manuscript figures/tables and their source outputs against the final manuscript; record any exceptions.
Once the missing MATLAB directories and documentation have been committed and verified, create GitHub release `v1.0.1` from the verified commit and archive that exact release on Zenodo. The initial `v1.0.0` archive was incomplete and must not be cited as the complete manuscript-matched code package.
Once the complete release has been archived, cite its version-specific Zenodo DOI in the manuscript alongside the GitHub repository URL. Use the concept DOI only for navigation across versions, and add the article DOI to the README when available.
GitHub's default branch may continue to change; the version-specific Zenodo archive should remain the stable reference for the implementation used in the published study.
J. Associated publication and citation
Manuscript: From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging  
Authors (manuscript order): Hoa Thi Pham; Sten Claessens; Ngoc-Tu Nguyen; Yangkang Yu; Sonny Pham; Martin Van Kranendonk  
Journal: Applied Mathematics and Computation  
Corresponding author: Hoa Thi Pham (`thihoa.pham@curtin.edu.au`; `pthoa.tdbd@hunre.edu.vn`)  
GitHub repository: https://github.com/HoaThiPham/RBF-Accuracy-to-Uncertainty  
Article DOI: To be added after assignment
If this research code is used or adapted, cite the associated research article and the version-specific Zenodo record of the code actually used. The research code is distributed under the accompanying Non-Commercial Academic Research Code License — Version 1.0, subject to the licence terms; copyright holder: Hoa Thi Pham. The code archive and restricted original geodetic data are governed by separate access and reuse conditions.

K. Research code archive and version identification
GitHub repository: https://github.com/HoaThiPham/RBF-Accuracy-to-Uncertainty  
