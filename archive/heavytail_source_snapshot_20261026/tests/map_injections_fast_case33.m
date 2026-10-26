% Map exported source-load injection scenarios to rectangular voltages with the
% fast power-to-voltage mapping (adaptive anchors, exact quadratic AC power map,
% fixed-Jacobian correction). Input and output files are given by environment
% variables FAST_MAP_IN and FAST_MAP_OUT (MAT files with P, Q in p.u., full bus).
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
BTMM_utils.io.setup_project();
inFile = string(getenv('FAST_MAP_IN')); outFile = string(getenv('FAST_MAP_OUT'));
assert(strlength(inFile) > 0 && strlength(outFile) > 0, 'TMM:FastMap:Files', 'Set FAST_MAP_IN and FAST_MAP_OUT.');
data = load(inFile);
caseName = "case33bw_modify";
tStart = tic;
atlas = BTMM_core.inverse.map_samples_with_atlas(caseName, data.P, data.Q);
mapSeconds = toc(tStart);
unresolved = find(~atlas.accepted);
state = atlas.fullState;
fallbackSeconds = 0;
if ~isempty(unresolved)
    tFallback = tic;
    define_constants; mpc = loadcase(char(caseName)); n = size(mpc.bus, 1);
    for idx = unresolved
        anchor = BTMM_core.powerflow.build_anchor(caseName, data.P(idx, :).', data.Q(idx, :).');
        state(:, idx) = [real(anchor.voltage); imag(anchor.voltage)];
    end
    fallbackSeconds = toc(tFallback);
end
anchorCount = atlas.anchorCount; acceptedCount = sum(atlas.accepted); maxPowerResidual = max(atlas.powerResidualInfinity(atlas.accepted));
save(outFile, 'state', 'anchorCount', 'acceptedCount', 'unresolved', 'mapSeconds', 'fallbackSeconds', 'maxPowerResidual', '-v7');
fprintf('[fast map] scenarios %d, anchors %d, accepted %d, AC fallback %d, map %.2f s, fallback %.2f s, max power residual %.3g p.u.\n', ...
    size(data.P, 1), anchorCount, acceptedCount, numel(unresolved), mapSeconds, fallbackSeconds, maxPowerResidual);
