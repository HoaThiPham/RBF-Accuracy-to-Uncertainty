Repository map
Associated article: From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging
Journal: Applied Mathematics and Computation (manuscript; publication details to be updated when available).
This map documents the source tree in the accompanying MATLAB code package. All paths below are relative to the repository root. The listings preserve the exact filename spelling, including the original `Boostrap`/`Bootstrap` naming variants, so that links correspond to the actual MATLAB files.
Repository entry point
`setup_project.m` — MATLAB setup helper at repository root. Read the setup instructions in `README.md` and `docs/REPRODUCTION_WORKFLOW.md` before running case-level scripts.
`README.md` — package overview, requirements, manuscript details, data-access conditions, and release information.
Code directories and script inventory
`+common/`
Shared numerical building blocks for kernel evaluation, RBF-system construction, shape-parameter handling, linear-system solution, prediction, bootstrap replicates, numerical-condition diagnostics, and uncertainty calibration.
`+common/applyRBFKernel.m`
`+common/bootstrapRBFReplicate.m`
`+common/buildEpsilon.m`
`+common/buildRBFSystem.m`
`+common/calculateConditionDiagnostics.m`
`+common/calibrateBootstrapVariance.m`
`+common/calibrateRBFVarianceTrainTest.m`
`+common/checkPolynomialAdmissibility.m`
`+common/evaluateBootstrapVarianceCalibration.m`
`+common/kernelHasShapeParameter.m`
`+common/MeanDistant.m`
`+common/predictRBF.m`
`+common/solveLinearSystem.m`
`+common/summarizeCalibrationAlphaBeta.m`
`+common/summarizeConditionDiagnostics.m`
`+globalrbf/`
Global-RBF numerical implementation: cross-validation, training/regularisation tuning, independent test prediction, ordinary and spatial bootstrap, variance calibration, and radial-segment resampling.
`+globalrbf/buildAdaptiveRadialSegments.m`
`+globalrbf/evaluateGlobalCV.m`
`+globalrbf/globalRBFBatchSummaryParallel_TuneReg.m`
`+globalrbf/globalRBFBootstrapBatchSummaryParallel.m`
`+globalrbf/globalRBFBootstrapForTesting.m`
`+globalrbf/globalRBFBootstrapForTraining.m`
`+globalrbf/globalRBFPredictTest.m`
`+globalrbf/globalRBFTrainingCV_TuneReg.m`
`+globalrbf/globalRBFVarianceCalibrationBatch.m`
`+globalrbf/resampleRadialSegments.m`
`+localrbf/`
Local-RBF numerical implementation: neighbourhood and pointwise optimisation, progressive Stage-1 checkpointing, radius/cap selection, train-to-test settings transfer, independent testing, bootstrap, and variance calibration.
`+localrbf/evaluateLocal.m`
`+localrbf/evaluateTargetExcludedError.m`
`+localrbf/finalizeStage1ProgressivePart.m`
`+localrbf/getNeighborhood.m`
`+localrbf/localRBFBatchSummaryParallel_TuneReg.m`
`+localrbf/localRBFBootstrapBatchSummaryParallel.m`
`+localrbf/localRBFBootstrapForTesting.m`
`+localrbf/localRBFBootstrapForTraining.m`
`+localrbf/LocalRBFTesting.m`
`+localrbf/LocalRBFTraining.m`
`+localrbf/localRBFVarianceCalibrationBatch.m`
`+localrbf/openStage1ProgressivePart.m`
`+localrbf/optimizeAtPointForAllCandidateRadii.m`
`+localrbf/optimizeParameters.m`
`+localrbf/saveStage1PartAtomic.m`
`+localrbf/selectBestRadiusCapWithToleranceOverTrainSet.m`
`+localrbf/selectBestRadiusWithToleranceAtEachRadiusCap.m`
`+localrbf/selectBestRadiusWithToleranceAtEachRadiusCapOverAllCaps.m`
`+localrbf/splitStage1PointOpt.m`
`+localrbf/stage1CheckpointPaths.m`
`+localrbf/transferSettingsFromTrainToTest.m`
`Run_Global_RBF/`
Global-RBF case-level production, ordinary/spatial bootstrap, and calibration entry scripts.
`Run_Global_RBF/Run_Global_RBF_Full_Production_Case1.m`
`Run_Global_RBF/Run_Global_RBF_Full_Production_Case2.m`
`Run_Global_RBF/Run_GlobalRBF_NonSpatialBootstrap.m`
`Run_Global_RBF/Run_GlobalRBF_SpatialBootstrap.m`
`Run_Global_RBF/RunGlobalCalibrationForNonSpatialBoostrapVariance.m`
`Run_Global_RBF/RunGlobalCalibrationForSpatialBoostrapVariance.m`
`Run_Local_RBF/Run_Main_Local/`
Local-RBF case-level production, bootstrap, and calibration entry scripts.
`Run_Local_RBF/Run_Main_Local/Run_Local_RBF_Bootstrap.m`
`Run_Local_RBF/Run_Main_Local/Run_Local_RBF_Full_Production_Case1.m`
`Run_Local_RBF/Run_Main_Local/Run_Local_RBF_Full_Production_Case2.m`
`Run_Local_RBF/Run_Main_Local/Run_Local_RBF_Variance_Calibration.m`
`Run_Local_RBF/Run_Check_Transfer_Sensitivity/`
Transfer-sensitivity runners for nearest-neighbour and inverse-distance-weighting (IDW) variants.
`Run_Local_RBF/Run_Check_Transfer_Sensitivity/Run_Local_RBF_Bootstrap_IDW.m`
`Run_Local_RBF/Run_Check_Transfer_Sensitivity/Run_Local_RBF_Bootstrap_Nearest.m`
`Run_Local_RBF/Run_Check_Transfer_Sensitivity/Run_Local_RBF_Full_Production_Case1_IDW.m`
`Run_Local_RBF/Run_Check_Transfer_Sensitivity/Run_Local_RBF_Full_Production_Case1_Nearest.m`
`Run_Local_RBF/Run_Check_Transfer_Sensitivity/Run_Local_RBF_Full_Production_Case2_IDW.m`
`Run_Local_RBF/Run_Check_Transfer_Sensitivity/Run_Local_RBF_Full_Production_Case2_Nearest.m`
`Run_Local_RBF/Run_Check_Transfer_Sensitivity/Run_Local_RBF_Variance_Calibration_IDW.m`
`Run_Local_RBF/Run_Check_Transfer_Sensitivity/Run_Local_RBF_Variance_Calibration_Nearest.m`
`Run_Local_RBF/Run_Check_Ablation_Sensitivity/`
Ablation and robustness runners comparing M0, MP, MR, and the full MPR framework.
`Run_Local_RBF/Run_Check_Ablation_Sensitivity/Run_Ablation_M0_Case.m`
`Run_Local_RBF/Run_Check_Ablation_Sensitivity/Run_Ablation_MP_Case.m`
`Run_Local_RBF/Run_Check_Ablation_Sensitivity/Run_Ablation_MR_Case.m`
`Run_Local_RBF/Run_Check_Ablation_Sensitivity/Run_Ablation_RandomRobustness_M0_MP_MR_vs_MPR.m`
`Fig_And_Table_RBF/`
Principal figure/table generation and saved-result post-processing scripts. Their input files depend on upstream production experiments.
`Fig_And_Table_RBF/Fig_Ablation_Fig1.m`
`Fig_And_Table_RBF/Fig_Compare_Calibration_ASE_RMSSE_FigC18C19.m`
`Fig_And_Table_RBF/Fig_TQ_Fig3.m`
`Fig_And_Table_RBF/FigAndTable_Radius_FigC5toC8_TableC1.m`
`Fig_And_Table_RBF/FigRBF_RMSE_ASE_RMSSE_Fig2_C9.m`
`Fig_And_Table_RBF/Figure_RBF_Variance_FigC10_C17.m`
`Fig_And_Table_RBF/Table_Ablation_C5.m`
`Fig_And_Table_RBF/Table_Calibration_AlphaBeta_Statistics_C6.m`
`Fig_And_Table_RBF/Table_ExtremeVariance_C7C8C9.m`
`Fig_And_Table_RBF/Table_Final_Hyperparameters_C2.m`
`Fig_And_Table_RBF/Table_Global_SpatialBootstrap_Sensitivity_C4.m`
`Fig_And_Table_RBF/Table_Model1_Technical_Setup_B1.m`
`Fig_And_Table_RBF/Table_NumericalCondition_C11C12.m`
`Fig_And_Table_RBF/Table_RunTime_C13.m`
`Fig_And_Table_RBF/Table_TQ_WeightSensitivity_C10.m`
`Fig_And_Table_RBF/Table_Transfer_Sensitivity_C3.m`
`Fig_And_Table_RBF/Support_Functions_For_Figures_And_Tables/`
Collector/helper functions used by figure and table scripts.
`Fig_And_Table_RBF/Support_Functions_For_Figures_And_Tables/Calculate_TQ_WeightSensitivity.m`
`Fig_And_Table_RBF/Support_Functions_For_Figures_And_Tables/Collect_Radius.m`
`Fig_And_Table_RBF/Support_Functions_For_Figures_And_Tables/Collect_RMSE_ASE_RMSSE.m`
`Fig_And_Table_RBF/Support_Functions_For_Figures_And_Tables/Collect_RMSE_ASE_RMSSE_IDW.m`
`Fig_And_Table_RBF/Support_Functions_For_Figures_And_Tables/Collect_RMSE_ASE_RMSSE_Nearest.m`
`Fig_And_Table_RBF/Support_Functions_For_Figures_And_Tables/Collect_SpatialVariance.m`
`Fig_And_Table_RBF/Support_Functions_For_Figures_And_Tables/Collect_TQ_Metrics_BeforeCalibration.m`
`docs/`
Repository and reproduction documentation.
`docs/REPOSITORY_MAP.md`
`docs/REPRODUCTION_WORKFLOW.md`
Reproduction boundaries
Original data not included: The original geodetic observations and prepared case inputs (`InputData_Case1.mat` and `InputData_Case2.mat`) are not present in this archive and are subject to the access policy stated in the manuscript. The file names are documented here as expected inputs, not as distributable datasets.
No automatic numerical verification implied: Listing a runner or exporter confirms that it exists in the source archive; it does not establish that MATLAB execution has succeeded or that the paper’s numerical results can be recreated without the restricted inputs and intermediate outputs.
Exact ordering and dependencies: See `docs/REPRODUCTION_WORKFLOW.md` for the actual production, calibration, sensitivity, ablation, and output sequence. Avoid running figure/table scripts before their required experiment results have been generated.
Archived research code: Use the version-specific Zenodo DOI of the code corresponding to the manuscript results, rather than the continuously changing GitHub default branch.
Research code package
This repository contains the MATLAB research code and reproducibility materials accompanying the article From Accuracy to Uncertainty: A Framework for RBF Interpolation Comparable to Kriging.
The package includes Global-RBF and Local-RBF implementations, parameter optimisation, bootstrap uncertainty calibration, sensitivity analyses, and scripts for generating the principal manuscript figures and tables.
Research Code Archive
GitHub repository:  
https://github.com/HoaThiPham/RBF-Accuracy-to-Uncertainty
Zenodo archive (all versions):  
https://doi.org/10.5281/zenodo.24375248
For reproducibility, please cite the version-specific DOI of the research code release corresponding to the published manuscript results.
