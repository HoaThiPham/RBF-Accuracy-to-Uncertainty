# Repository map

**Associated article:** *From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging*

**Journal:** *Applied Mathematics and Computation*

This map documents the source tree in the accompanying MATLAB code package. All paths below are relative to the repository root. The listings preserve the exact filename spelling, including the original `Boostrap`/`Bootstrap` naming variants, so that links correspond to the actual MATLAB files.

## Repository entry point

* `setup\\\\\\\_project.m` — MATLAB setup helper at repository root. Read the setup instructions in `README.md` and `docs/REPRODUCTION\\\\\\\_WORKFLOW.md` before running case-level scripts.
* `README.md` — package overview, requirements, manuscript details, data-access conditions, and release information.

## Code directories and script inventory

### `+common/`

Shared numerical building blocks for kernel evaluation, RBF-system construction, shape-parameter handling, linear-system solution, prediction, bootstrap replicates, numerical-condition diagnostics, and uncertainty calibration.

* `+common/applyRBFKernel.m`
* `+common/bootstrapRBFReplicate.m`
* `+common/buildEpsilon.m`
* `+common/buildRBFSystem.m`
* `+common/calculateConditionDiagnostics.m`
* `+common/calibrateBootstrapVariance.m`
* `+common/calibrateRBFVarianceTrainTest.m`
* `+common/checkPolynomialAdmissibility.m`
* `+common/evaluateBootstrapVarianceCalibration.m`
* `+common/kernelHasShapeParameter.m`
* `+common/MeanDistant.m`
* `+common/predictRBF.m`
* `+common/solveLinearSystem.m`
* `+common/summarizeCalibrationAlphaBeta.m`
* `+common/summarizeConditionDiagnostics.m`

### `+globalrbf/`

Global-RBF numerical implementation: cross-validation, training/regularisation tuning, independent test prediction, ordinary and spatial bootstrap, variance calibration, and radial-segment resampling.

* `+globalrbf/buildAdaptiveRadialSegments.m`
* `+globalrbf/evaluateGlobalCV.m`
* `+globalrbf/globalRBFBatchSummaryParallel\\\\\\\_TuneReg.m`
* `+globalrbf/globalRBFBootstrapBatchSummaryParallel.m`
* `+globalrbf/globalRBFBootstrapForTesting.m`
* `+globalrbf/globalRBFBootstrapForTraining.m`
* `+globalrbf/globalRBFPredictTest.m`
* `+globalrbf/globalRBFTrainingCV\\\\\\\_TuneReg.m`
* `+globalrbf/globalRBFVarianceCalibrationBatch.m`
* `+globalrbf/resampleRadialSegments.m`

### `+localrbf/`

Local-RBF numerical implementation: neighbourhood and pointwise optimisation, progressive Stage-1 checkpointing, radius/cap selection, train-to-test settings transfer, independent testing, bootstrap, and variance calibration.

* `+localrbf/evaluateLocal.m`
* `+localrbf/evaluateTargetExcludedError.m`
* `+localrbf/finalizeStage1ProgressivePart.m`
* `+localrbf/getNeighborhood.m`
* `+localrbf/localRBFBatchSummaryParallel\\\\\\\_TuneReg.m`
* `+localrbf/localRBFBootstrapBatchSummaryParallel.m`
* `+localrbf/localRBFBootstrapForTesting.m`
* `+localrbf/localRBFBootstrapForTraining.m`
* `+localrbf/LocalRBFTesting.m`
* `+localrbf/LocalRBFTraining.m`
* `+localrbf/localRBFVarianceCalibrationBatch.m`
* `+localrbf/openStage1ProgressivePart.m`
* `+localrbf/optimizeAtPointForAllCandidateRadii.m`
* `+localrbf/optimizeParameters.m`
* `+localrbf/saveStage1PartAtomic.m`
* `+localrbf/selectBestRadiusCapWithToleranceOverTrainSet.m`
* `+localrbf/selectBestRadiusWithToleranceAtEachRadiusCap.m`
* `+localrbf/selectBestRadiusWithToleranceAtEachRadiusCapOverAllCaps.m`
* `+localrbf/splitStage1PointOpt.m`
* `+localrbf/stage1CheckpointPaths.m`
* `+localrbf/transferSettingsFromTrainToTest.m`

### `Run\\\\\\\_Global\\\\\\\_RBF/`

Global-RBF case-level production, ordinary/spatial bootstrap, and calibration entry scripts.

* `Run\\\\\\\_Global\\\\\\\_RBF/Run\\\\\\\_Global\\\\\\\_RBF\\\\\\\_Full\\\\\\\_Production\\\\\\\_Case1.m`
* `Run\\\\\\\_Global\\\\\\\_RBF/Run\\\\\\\_Global\\\\\\\_RBF\\\\\\\_Full\\\\\\\_Production\\\\\\\_Case2.m`
* `Run\\\\\\\_Global\\\\\\\_RBF/Run\\\\\\\_GlobalRBF\\\\\\\_NonSpatialBootstrap.m`
* `Run\\\\\\\_Global\\\\\\\_RBF/Run\\\\\\\_GlobalRBF\\\\\\\_SpatialBootstrap.m`
* `Run\\\\\\\_Global\\\\\\\_RBF/RunGlobalCalibrationForNonSpatialBoostrapVariance.m`
* `Run\\\\\\\_Global\\\\\\\_RBF/RunGlobalCalibrationForSpatialBoostrapVariance.m`

### `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Main\\\\\\\_Local/`

Local-RBF case-level production, bootstrap, and calibration entry scripts.

* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Main\\\\\\\_Local/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Bootstrap.m`
* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Main\\\\\\\_Local/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Full\\\\\\\_Production\\\\\\\_Case1.m`
* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Main\\\\\\\_Local/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Full\\\\\\\_Production\\\\\\\_Case2.m`
* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Main\\\\\\\_Local/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Variance\\\\\\\_Calibration.m`

### `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Transfer\\\\\\\_Sensitivity/`

Transfer-sensitivity runners for nearest-neighbour and inverse-distance-weighting (IDW) variants.

* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Transfer\\\\\\\_Sensitivity/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Bootstrap\\\\\\\_IDW.m`
* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Transfer\\\\\\\_Sensitivity/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Bootstrap\\\\\\\_Nearest.m`
* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Transfer\\\\\\\_Sensitivity/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Full\\\\\\\_Production\\\\\\\_Case1\\\\\\\_IDW.m`
* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Transfer\\\\\\\_Sensitivity/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Full\\\\\\\_Production\\\\\\\_Case1\\\\\\\_Nearest.m`
* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Transfer\\\\\\\_Sensitivity/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Full\\\\\\\_Production\\\\\\\_Case2\\\\\\\_IDW.m`
* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Transfer\\\\\\\_Sensitivity/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Full\\\\\\\_Production\\\\\\\_Case2\\\\\\\_Nearest.m`
* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Transfer\\\\\\\_Sensitivity/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Variance\\\\\\\_Calibration\\\\\\\_IDW.m`
* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Transfer\\\\\\\_Sensitivity/Run\\\\\\\_Local\\\\\\\_RBF\\\\\\\_Variance\\\\\\\_Calibration\\\\\\\_Nearest.m`

### `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Ablation\\\\\\\_Sensitivity/`

Ablation and robustness runners comparing M0, MP, MR, and the full MPR framework.

* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Ablation\\\\\\\_Sensitivity/Run\\\\\\\_Ablation\\\\\\\_M0\\\\\\\_Case.m`
* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Ablation\\\\\\\_Sensitivity/Run\\\\\\\_Ablation\\\\\\\_MP\\\\\\\_Case.m`
* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Ablation\\\\\\\_Sensitivity/Run\\\\\\\_Ablation\\\\\\\_MR\\\\\\\_Case.m`
* `Run\\\\\\\_Local\\\\\\\_RBF/Run\\\\\\\_Check\\\\\\\_Ablation\\\\\\\_Sensitivity/Run\\\\\\\_Ablation\\\\\\\_RandomRobustness\\\\\\\_M0\\\\\\\_MP\\\\\\\_MR\\\\\\\_vs\\\\\\\_MPR.m`

### `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/`

Principal figure/table generation and saved-result post-processing scripts. Their input files depend on upstream production experiments.

* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Fig\\\\\\\_Ablation\\\\\\\_Fig1.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Fig\\\\\\\_Compare\\\\\\\_Calibration\\\\\\\_ASE\\\\\\\_RMSSE\\\\\\\_FigC18C19.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Fig\\\\\\\_TQ\\\\\\\_Fig3.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/FigAndTable\\\\\\\_Radius\\\\\\\_FigC5toC8\\\\\\\_TableC1.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/FigRBF\\\\\\\_RMSE\\\\\\\_ASE\\\\\\\_RMSSE\\\\\\\_Fig2\\\\\\\_C9.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Figure\\\\\\\_RBF\\\\\\\_Variance\\\\\\\_FigC10\\\\\\\_C17.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Table\\\\\\\_Ablation\\\\\\\_C5.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Table\\\\\\\_Calibration\\\\\\\_AlphaBeta\\\\\\\_Statistics\\\\\\\_C6.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Table\\\\\\\_ExtremeVariance\\\\\\\_C7C8C9.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Table\\\\\\\_Final\\\\\\\_Hyperparameters\\\\\\\_C2.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Table\\\\\\\_Global\\\\\\\_SpatialBootstrap\\\\\\\_Sensitivity\\\\\\\_C4.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Table\\\\\\\_Model1\\\\\\\_Technical\\\\\\\_Setup\\\\\\\_B1.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Table\\\\\\\_NumericalCondition\\\\\\\_C11C12.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Table\\\\\\\_RunTime\\\\\\\_C13.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Table\\\\\\\_TQ\\\\\\\_WeightSensitivity\\\\\\\_C10.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Table\\\\\\\_Transfer\\\\\\\_Sensitivity\\\\\\\_C3.m`

### `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Support\\\\\\\_Functions\\\\\\\_For\\\\\\\_Figures\\\\\\\_And\\\\\\\_Tables/`

Collector/helper functions used by figure and table scripts.

* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Support\\\\\\\_Functions\\\\\\\_For\\\\\\\_Figures\\\\\\\_And\\\\\\\_Tables/Calculate\\\\\\\_TQ\\\\\\\_WeightSensitivity.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Support\\\\\\\_Functions\\\\\\\_For\\\\\\\_Figures\\\\\\\_And\\\\\\\_Tables/Collect\\\\\\\_Radius.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Support\\\\\\\_Functions\\\\\\\_For\\\\\\\_Figures\\\\\\\_And\\\\\\\_Tables/Collect\\\\\\\_RMSE\\\\\\\_ASE\\\\\\\_RMSSE.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Support\\\\\\\_Functions\\\\\\\_For\\\\\\\_Figures\\\\\\\_And\\\\\\\_Tables/Collect\\\\\\\_RMSE\\\\\\\_ASE\\\\\\\_RMSSE\\\\\\\_IDW.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Support\\\\\\\_Functions\\\\\\\_For\\\\\\\_Figures\\\\\\\_And\\\\\\\_Tables/Collect\\\\\\\_RMSE\\\\\\\_ASE\\\\\\\_RMSSE\\\\\\\_Nearest.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Support\\\\\\\_Functions\\\\\\\_For\\\\\\\_Figures\\\\\\\_And\\\\\\\_Tables/Collect\\\\\\\_SpatialVariance.m`
* `Fig\\\\\\\_And\\\\\\\_Table\\\\\\\_RBF/Support\\\\\\\_Functions\\\\\\\_For\\\\\\\_Figures\\\\\\\_And\\\\\\\_Tables/Collect\\\\\\\_TQ\\\\\\\_Metrics\\\\\\\_BeforeCalibration.m`

### `docs/`

Repository and reproduction documentation.

* `docs/REPOSITORY\\\\\\\_MAP.md`
* `docs/REPRODUCTION\\\\\\\_WORKFLOW.md`

## Reproduction boundaries

* **Original data not included:** The original geodetic observations and prepared case inputs (`InputData\\\\\\\_Case1.mat` and `InputData\\\\\\\_Case2.mat`) are not present in this archive and are subject to the access policy stated in the manuscript. The file names are documented here as expected inputs, not as distributable datasets.
* **No automatic numerical verification implied:** Listing a runner or exporter confirms that it exists in the source archive; it does not establish that MATLAB execution has succeeded or that the paper’s numerical results can be recreated without the restricted inputs and intermediate outputs.
* **Exact ordering and dependencies:** See `docs/REPRODUCTION\\\\\\\_WORKFLOW.md` for the actual production, calibration, sensitivity, ablation, and output sequence. Avoid running figure/table scripts before their required experiment results have been generated.
* **Archived research code:** Use the version-specific Zenodo DOI of the code corresponding to the manuscript results, rather than the continuously changing GitHub default branch.

## Research code package

This repository contains the MATLAB research code and reproducibility materials accompanying the article *From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging*.

The package includes Global-RBF and Local-RBF implementations, parameter optimisation, bootstrap uncertainty calibration, sensitivity analyses, and scripts for generating the principal manuscript figures and tables.

## Research Code Archive

**GitHub repository:**  
https://github.com/HoaThiPham/RBF-Accuracy-to-Uncertainty

For reproducibility, please cite the version-specific DOI of the research code release corresponding to the published manuscript results.

