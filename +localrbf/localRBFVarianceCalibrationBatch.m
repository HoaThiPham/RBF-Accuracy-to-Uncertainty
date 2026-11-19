function Results = localRBFVarianceCalibrationBatch( ...
    FitResults, BootstrapResults, opts, OutputFile)
%LOCALRBFVARIANCECALIBRATIONBATCH Calibrate local-RBF bootstrap variance.
%
% alpha/beta are estimated ONLY from the training residual vector paired
% with training bootstrap variance. The same alpha/beta are then applied
% to the independent test residual/variance. Training residual provenance
% is taken from BootstrapResults.Run.<run>.Train.Reference so that
% 'original' and 'pseudo-unseen' training bootstrap modes cannot be mixed
% accidentally.
%
% Calibration.Run uses the canonical Model_Kernel field naming used by the
% common calibration-summary functions. The detailed Local-RBF source run
% names are preserved separately in SourceRunFieldMatrix, so no provenance
% information is lost and common functions do not need Local-specific
% filename/run-name knowledge.
%
% This mirrors globalrbf.globalRBFVarianceCalibrationBatch and reuses:
%   common.calibrateRBFVarianceTrainTest

if nargin<3||isempty(opts), opts=struct(); end
if nargin<4, OutputFile=''; end
if isfield(FitResults,'Results')&&isstruct(FitResults.Results)&&isscalar(FitResults.Results), FitResults=FitResults.Results; end
if isfield(BootstrapResults,'Results')&&isstruct(BootstrapResults.Results)&&isscalar(BootstrapResults.Results), BootstrapResults=BootstrapResults.Results; end

req={'CaseName','ListOfModel','NameFunction_RBF','Run'};
for k=1:numel(req), if ~isfield(BootstrapResults,req{k}), error('BootstrapResults.%s is required.',req{k}); end, end
ListOfModel=BootstrapResults.ListOfModel; NameFunction_RBF=BootstrapResults.NameFunction_RBF;
if isfield(FitResults,'ListOfModel')&&~isequal(cellstr(string(FitResults.ListOfModel)),cellstr(string(ListOfModel))), error('FitResults and BootstrapResults model lists differ.'); end
if isfield(FitResults,'NameFunction_RBF')&&~isequal(cellstr(string(FitResults.NameFunction_RBF)),cellstr(string(NameFunction_RBF))), error('FitResults and BootstrapResults kernel lists differ.'); end
nModel=numel(ListOfModel); nKernel=numel(NameFunction_RBF); nRun=nModel*nKernel;

Results=struct(); Results.CaseName=BootstrapResults.CaseName; Results.ListOfModel=ListOfModel; Results.NameFunction_RBF=NameFunction_RBF;
Results.Settings=opts; if isfield(BootstrapResults,'Settings')&&isfield(BootstrapResults.Settings,'TrainingMode'), Results.Settings.TrainingMode=BootstrapResults.Settings.TrainingMode; end
Results.MatrixInfo.rows=ListOfModel; Results.MatrixInfo.columns=NameFunction_RBF; Results.MatrixInfo.description='Rows = input models; columns = RBF kernels';
fields={'alpha','beta','Loss','Train_ASE_RMSE_Pre','Train_RMSSE_Pre','Train_ASE_RMSE','Train_RMSSE','Test_ASE_RMSE_Pre','Test_RMSSE_Pre','Test_ASE_RMSE','Test_RMSSE','wallTime'};
for k=1:numel(fields), Results.(fields{k})=nan(nModel,nKernel); end
Results.success=false(nModel,nKernel); Results.message=cell(nModel,nKernel); Results.Run=struct();

% Canonical calibration run mapping used by common summary functions.
Results.RunFieldMatrix=cell(nModel,nKernel);

% Original detailed Local-RBF bootstrap run mapping retained for provenance.
Results.SourceRunFieldMatrix=cell(nModel,nKernel);

pool=gcp('nocreate'); if isempty(pool), pool=parpool; end
Results.Settings.Parallel=true; Results.Settings.NumWorkers=pool.NumWorkers;
RunCell=cell(nRun,1); tParallel=tic;

parfor irun=1:nRun
    i=floor((irun-1)/nKernel)+1; j=mod(irun-1,nKernel)+1;
    modelName=char(ListOfModel{i}); kernel=char(NameFunction_RBF{j});
    sourceRunName=resolveBootstrapRunNameLocal(BootstrapResults,modelName,kernel,i,j);
    calibrationRunName=[modelName '_' kernel];
    if ~isvarname(calibrationRunName)
        error('Canonical calibration run name is not a valid MATLAB field: %s',calibrationRunName);
    end
    R=struct('i',i,'j',j,'sourceRunName',sourceRunName,'calibrationRunName',calibrationRunName, ...
        'success',false,'message','','Calibration',[],'wallTime',NaN);
    t=tic;
    try
        if ~isfield(BootstrapResults.Run,sourceRunName), error('BootstrapResults.Run.%s was not found.',sourceRunName); end
        BR=BootstrapResults.Run.(sourceRunName);
        if ~isfield(BR,'Train')||isempty(BR.Train)||~isfield(BR,'Test')||isempty(BR.Test), error('Bootstrap train/test results are incomplete.'); end
        eTrain=getReferenceResidualLocal(BR.Train,'training');
        eTest=getReferenceResidualLocal(BR.Test,'testing');
        vTrain=BR.Train.Variance(:); vTest=BR.Test.Variance(:);
        if numel(eTrain)~=numel(vTrain), error('Training residual/variance size mismatch.'); end
        if numel(eTest)~=numel(vTest), error('Testing residual/variance size mismatch.'); end
        R.Calibration=common.calibrateRBFVarianceTrainTest(eTrain,vTrain,eTest,vTest,opts);
        R.success=R.Calibration.success;
    catch ME
        R.success=false; R.message=ME.message;
    end
    R.wallTime=toc(t); RunCell{irun}=R;
end
Results.parallelWallTime=toc(tParallel);

for irun=1:nRun
    R=RunCell{irun}; i=R.i; j=R.j;

    % Mapping is retained even when a calibration run fails.
    Results.RunFieldMatrix{i,j}=R.calibrationRunName;
    Results.SourceRunFieldMatrix{i,j}=R.sourceRunName;

    Results.success(i,j)=R.success; Results.message{i,j}=R.message; Results.wallTime(i,j)=R.wallTime;
    if ~R.success, continue, end

    C=R.Calibration; Results.alpha(i,j)=C.alpha; Results.beta(i,j)=C.beta; Results.Loss(i,j)=C.Loss;
    Results.Train_ASE_RMSE_Pre(i,j)=C.Train.ASE_RMSE_Pre; Results.Train_RMSSE_Pre(i,j)=C.Train.RMSSE_Pre;
    Results.Train_ASE_RMSE(i,j)=C.Train.ASE_RMSE; Results.Train_RMSSE(i,j)=C.Train.RMSSE;
    Results.Test_ASE_RMSE_Pre(i,j)=C.Test.ASE_RMSE_Pre; Results.Test_RMSSE_Pre(i,j)=C.Test.RMSSE_Pre;
    Results.Test_ASE_RMSE(i,j)=C.Test.ASE_RMSE; Results.Test_RMSSE(i,j)=C.Test.RMSSE;

    % Store one copy only, under the canonical name expected by common
    % summary functions. The original source name is retained in the
    % SourceRunFieldMatrix metadata above.
    Results.Run.(R.calibrationRunName)=C;
end

if ~isempty(OutputFile)
    d=fileparts(OutputFile); if ~isempty(d)&&~exist(d,'dir'), mkdir(d); end
    save(OutputFile,'Results','-v7.3');
end
end

function runName=resolveBootstrapRunNameLocal(B,model,kernel,i,j)
runName='';
if isfield(B,'RunFieldMatrix')&&iscell(B.RunFieldMatrix)&&size(B.RunFieldMatrix,1)>=i&&size(B.RunFieldMatrix,2)>=j
    runName=B.RunFieldMatrix{i,j};
end
if isempty(runName)
    candidate=[model '_' kernel];
    if isfield(B.Run,candidate), runName=candidate; else
        fn=fieldnames(B.Run); hit='';
        for q=1:numel(fn)
            Z=B.Run.(fn{q});
            if isstruct(Z)&&isfield(Z,'model')&&isfield(Z,'kernel')&&strcmpi(char(string(Z.model)),model)&&strcmpi(char(string(Z.kernel)),kernel), hit=fn{q}; break, end
        end
        if isempty(hit), error('Cannot resolve bootstrap run for %s | %s.',model,kernel); end
        runName=hit;
    end
end
runName=char(string(runName));
end

function e=getReferenceResidualLocal(B,label)
if ~isstruct(B)||~isfield(B,'Reference')||~isstruct(B.Reference)||~isfield(B.Reference,'Residual')
    error('Bootstrap %s Reference.Residual is required for calibration.',label);
end
e=B.Reference.Residual(:); if isempty(e)||any(~isfinite(e)), error('Bootstrap %s Reference.Residual contains missing/nonfinite values.',label); end
end
