function runSummary = BTMM_case141(projectRoot)
%% BTMM_case141
% Formal case workflow for the case141 network.
%
% This root entry orchestrates the full case141 formal workflow. Package
% functions implement data loading, proposed-method computation, baselines,
% metrics, output writing, and figure export. PROJECTROOT is optional and is
% used to isolate contract tests from the working tree.
%
% Workflow steps:
% 1. Define case configuration.
% 2. Prepare directories, seed, and run state.
% 3. Load and validate data inputs.
% 4. Run the proposed method.
% 5. Run SOTA or baseline methods.
% 6. Evaluate case metrics.
% 7. Write tables and summary.
% 8. Export formal case figures.
% 9. Finalize the run, write run manifest, and report evidence registration status.

if nargin < 1
    projectRoot = string(fileparts(mfilename("fullpath")));
else
    projectRoot = string(projectRoot);
    if ~isscalar(projectRoot) || strlength(projectRoot) == 0
        error("BTMM:case141:InvalidProjectRoot", ...
            "projectRoot must be a nonempty scalar path.");
    end
end

caseConfig = local_case_config(projectRoot);
tracker = BTMM_utils.workflow.create_step_tracker(9, caseConfig.name);
trackerCleanup = onCleanup(@() tracker.close("aborted"));

runContext = struct();
runContext.invokedAt = string(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss"));
runContext.runState = struct();
runContext.caseData = struct();
runContext.proposedResults = struct();
runContext.baselineResults = struct();
runContext.metrics = struct();
runContext.outputFiles = struct();
runContext.figureOutputs = struct();
failedStage = struct("index", 0, "name", "Initialize case workflow");

try
    failedStage = struct("index", 1, "name", "Define case configuration");
    tracker.update(failedStage.index, failedStage.name);

    failedStage = struct("index", 2, "name", "Prepare directories, seed, and run state");
    tracker.update(failedStage.index, failedStage.name);
    runState = BTMM_utils.workflow.prepare_case_run(caseConfig);
    runContext.runState = runState;

    failedStage = struct("index", 3, "name", "Load and validate data inputs");
    tracker.update(failedStage.index, failedStage.name);
    caseData = BTMM_utils.io.load_case_inputs(caseConfig);
    runContext.caseData = caseData;
    BTMM_utils.workflow.assert_stage_ready( ...
        caseData, failedStage.index, failedStage.name);

    failedStage = struct("index", 4, "name", "Run the proposed method");
    tracker.update(failedStage.index, failedStage.name);
    proposedResults = BTMM_core.methods.run_proposed_case(caseConfig, caseData);
    runContext.proposedResults = proposedResults;
    BTMM_utils.workflow.assert_stage_ready( ...
        proposedResults, failedStage.index, failedStage.name);

    failedStage = struct("index", 5, "name", "Run SOTA or baseline methods");
    tracker.update(failedStage.index, failedStage.name);
    baselineResults = BTMM_sota.baselines.run_case_baselines( ...
        caseConfig, caseData, proposedResults);
    runContext.baselineResults = baselineResults;
    BTMM_utils.workflow.assert_stage_ready( ...
        baselineResults, failedStage.index, failedStage.name);

    failedStage = struct("index", 6, "name", "Evaluate case metrics");
    tracker.update(failedStage.index, failedStage.name);
    metrics = BTMM_core.metrics.evaluate_case_metrics( ...
        caseConfig, proposedResults, baselineResults);
    runContext.metrics = metrics;
    BTMM_utils.workflow.assert_stage_ready( ...
        metrics, failedStage.index, failedStage.name);

    failedStage = struct("index", 7, "name", "Write tables and summary");
    tracker.update(failedStage.index, failedStage.name);
    outputFiles = BTMM_utils.io.write_case_outputs( ...
        caseConfig, caseData, proposedResults, baselineResults, metrics);
    runContext.outputFiles = outputFiles;
    BTMM_utils.workflow.assert_stage_ready( ...
        outputFiles, failedStage.index, failedStage.name);

    failedStage = struct("index", 8, "name", "Export formal case figures");
    tracker.update(failedStage.index, failedStage.name);
    figureOutputs = BTMM_utils.plotting.export_case_figures(caseConfig, metrics);
    runContext.figureOutputs = figureOutputs;
    BTMM_utils.workflow.assert_stage_ready( ...
        figureOutputs, failedStage.index, failedStage.name);

    failedStage = struct("index", 9, "name", ...
        "Finalize run, write manifest, and report evidence registration status");
    tracker.update(failedStage.index, failedStage.name);
    runSummary = BTMM_utils.workflow.finalize_case_run( ...
        caseConfig, runState, caseData, proposedResults, baselineResults, ...
        metrics, outputFiles, figureOutputs);
    BTMM_utils.workflow.assert_stage_ready( ...
        runSummary, failedStage.index, failedStage.name);

    fprintf("Evidence reminder: register %s in 01_IDEA/evidence_map.md when results become formal.\n", ...
        caseConfig.evidenceId);
    tracker.complete();
    clear trackerCleanup
catch runError
    tracker.fail(runError);
    try
        BTMM_utils.workflow.finalize_failed_case_run( ...
            caseConfig, failedStage, runError, runContext);
    catch manifestError
        runError = addCause(runError, manifestError);
    end
    clear trackerCleanup
    rethrow(runError)
end
end

function caseConfig = local_case_config(projectRoot)
caseConfig = struct();
caseConfig.projectRoot = string(projectRoot);
caseConfig.name = "case141";
caseConfig.network = "case141";
caseConfig.description = "Proposed method and registered baselines on the case141 network";
caseConfig.entryFunction = "BTMM_case141";
caseConfig.seed = 123;
caseConfig.dataDir = fullfile(projectRoot, "data");
caseConfig.cacheDir = fullfile(projectRoot, "cache", caseConfig.name);
caseConfig.resultDir = fullfile(projectRoot, "result", caseConfig.name);
caseConfig.figureDir = fullfile(caseConfig.resultDir, "figures");
caseConfig.claim = "C2";
caseConfig.evidenceId = "E02";
caseConfig.contractVersion = "2026.07.12";
caseConfig.lifecycleStage = "one-to-hundred-evidence";
caseConfig.caseContract = "new-method";
caseConfig.figurePlanId = "FP-C2";
caseConfig.optimizerPriority = ["cvx_mosek", "gurobi", "mosek_direct"];
caseConfig.showTaskWaitbar = usejava("desktop");
caseConfig.workflowStatus = "scaffold";
end
