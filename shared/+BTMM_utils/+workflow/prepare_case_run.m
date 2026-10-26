function runState = prepare_case_run(caseConfig)
%PREPARE_CASE_RUN Prepare directories, seed, and scaffold run state.

requiredFields = ["name", "dataDir", "cacheDir", "resultDir", "figureDir", "seed"];
for k = 1:numel(requiredFields)
    fieldName = requiredFields(k);
    if ~isfield(caseConfig, fieldName)
        error("BTMM_utils:workflow:MissingConfigField", ...
            "caseConfig.%s is required.", fieldName);
    end
end

BTMM_utils.io.ensure_dir(caseConfig.cacheDir);
BTMM_utils.io.ensure_dir(caseConfig.resultDir);
BTMM_utils.io.ensure_dir(caseConfig.figureDir);

rngState = BTMM_utils.mc.init_seed(caseConfig.seed);

runState = struct();
runState.caseName = string(caseConfig.name);
runState.status = "scaffold";
runState.startedAt = string(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss"));
runState.rngSeed = caseConfig.seed;
runState.rngType = string(rngState.Type);
runState.resultDir = string(caseConfig.resultDir);
runState.figureDir = string(caseConfig.figureDir);
runState.cacheDir = string(caseConfig.cacheDir);
runState.files = struct();
runState.warnings = strings(0, 1);
end
