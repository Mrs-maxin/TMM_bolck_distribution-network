function atlasResult = map_samples_with_atlas(caseName, samplesP, samplesQ, options)
%MAP_SAMPLES_WITH_ATLAS Recover rectangular voltage samples with adaptive anchors.
%   Each anchor costs one AC power flow and one Jacobian factorization. All
%   samples are then mapped in a batch by the exact quadratic AC power map.

arguments
    caseName (1, 1) string
    samplesP (:, :) double
    samplesQ (:, :) double
    options.PowerFlowTolerance (1,1) double {mustBePositive} = 1e-8
    options.MaximumAnchorCount (1, 1) double {mustBeInteger, mustBePositive} = 12
    options.StateTolerance (1, 1) double {mustBePositive} = 1e-7
    options.MaximumCorrections (1,1) double {mustBeInteger,mustBeNonnegative} = 10
    options.PowerTolerance (1,1) double {mustBePositive} = 1e-8
    options.VoltageMin (1, 1) double {mustBePositive} = 0.80
    options.VoltageMax (1, 1) double {mustBePositive} = 1.20
end

assert(isequal(size(samplesP), size(samplesQ)), ...
    "BTMM:inverse:SampleDimension", ...
    "samplesP and samplesQ must have identical dimensions.");

tStart = tic;
sampleCount = size(samplesP, 1);
busCount = size(samplesP, 2);
baseCase = loadcase(char(caseName));
define_constants;
slack = find(baseCase.bus(:, BUS_TYPE) == REF, 1);
assert(isscalar(find(baseCase.bus(:,BUS_TYPE)==REF)),"BTMM:inverse:SlackBus", ...
    "One fixed slack bus is required.");
pq = setdiff((1:busCount).', slack, "stable");
assert(busCount==size(baseCase.bus,1) && all(baseCase.bus(pq,BUS_TYPE)==PQ) && ...
    all(baseCase.gen(baseCase.gen(:,GEN_STATUS)>0,GEN_BUS)==slack), ...
    "BTMM:inverse:BusTypes","Fixed PQ buses and slack-only generation are required.");
assert(sampleCount>0 && all(isfinite(samplesP),'all') && all(isfinite(samplesQ),'all'));
targetPower = [samplesP(:, pq).'; samplesQ(:, pq).'];

sampleCenter = mean(targetPower, 2);
[~, firstIndex] = min(sum((targetPower - sampleCenter).^2, 1));

stateCount = 2 * numel(pq);
bestState = nan(stateCount, sampleCount);
bestScore = inf(1, sampleCount);
bestResidual = inf(1, sampleCount);
bestBound = inf(1, sampleCount);
bestAnchor = zeros(1, sampleCount);
bestInfinity = inf(1,sampleCount);
bestCorrection = inf(1,sampleCount);
correctionCount = zeros(1,sampleCount);
totalCorrectionCount = zeros(1,sampleCount);
backtrackCount = zeros(1,sampleCount);
accepted = false(1,sampleCount);
predictorState = nan(stateCount,sampleCount);
predictorScore = inf(1,sampleCount);
anchorIndices = zeros(options.MaximumAnchorCount, 1);
anchors = cell(options.MaximumAnchorCount, 1);

nextIndex = firstIndex;
anchorCount = 0;
for anchorNumber = 1:options.MaximumAnchorCount
    anchorCount = anchorNumber;
    anchorIndices(anchorNumber) = nextIndex;
    anchor = BTMM_core.powerflow.build_anchor( ...
        caseName, samplesP(nextIndex, :).', samplesQ(nextIndex, :).', ...
        PowerFlowTolerance=options.PowerFlowTolerance);
    anchors{anchorNumber} = anchor;

    inverseAction = @(rhs) anchor.inverseMatrix * rhs;
    powerMap = @(state) BTMM_core.powerflow.evaluate_reduced_power(state, anchor);
    active = find(~accepted);
    mapped = BTMM_core.inverse.map_power_samples_quadratic( ...
        anchor.state, anchor.power, targetPower(:,active), inverseAction, powerMap);
    atAnchor = find(active==nextIndex);
    mapped.state(:,atAnchor) = anchor.state;
    mapped.correctedResidual = targetPower(:,active)-powerMap(mapped.state);
    initialScore = anchor.inverseNorm*vecnorm(mapped.correctedResidual,2,1);
    initialImprove = initialScore<predictorScore(active);
    predictorState(:,active(initialImprove)) = mapped.state(:,initialImprove);
    predictorScore(active(initialImprove)) = initialScore(initialImprove);
    refined = BTMM_core.inverse.refine_fixed_jacobian(mapped.state,targetPower(:,active),anchor, ...
        MaximumCorrections=options.MaximumCorrections,StateTolerance=options.StateTolerance, ...
        PowerTolerance=options.PowerTolerance,VoltageMin=options.VoltageMin,VoltageMax=options.VoltageMax);
    mapped.state = refined.state;
    mapped.correctedResidual = refined.residual;
    totalCorrectionCount(active) = totalCorrectionCount(active)+refined.correctionCount;
    backtrackCount(active) = backtrackCount(active)+refined.backtrackCount;

    residualNorm = vecnorm(mapped.correctedResidual, 2, 1);
    stateResidual = anchor.inverseNorm * residualNorm;
    displacement = vecnorm(mapped.state - anchor.state, 2, 1);
    contraction = 2 * anchor.inverseNorm * anchor.admittanceNorm .* displacement;
    localBound = inf(1, numel(active));
    validContraction = contraction < 1;
    localBound(validContraction) = stateResidual(validContraction) ./ ...
        max(1 - contraction(validContraction), eps);

    npq = numel(pq);
    magnitude = hypot(mapped.state(1:npq, :), mapped.state(npq + (1:npq), :));
    voltageInside = all(magnitude >= options.VoltageMin & ...
        magnitude <= options.VoltageMax, 1);
    score = stateResidual;
    score(~voltageInside | ~isfinite(score)) = inf;

    improve = score < bestScore(active) | refined.accepted;
    selected = active(improve);
    bestState(:, selected) = mapped.state(:, improve);
    bestScore(selected) = score(improve);
    bestResidual(selected) = residualNorm(improve);
    bestBound(selected) = localBound(improve);
    bestAnchor(selected) = anchorNumber;
    bestInfinity(selected) = refined.powerResidualInfinity(improve);
    bestCorrection(selected) = refined.correctionNorm(improve);
    correctionCount(selected) = refined.correctionCount(improve);
    accepted(selected) = refined.accepted(improve);

    if all(accepted)
        break
    end

    used = anchorIndices(1:anchorNumber);
    rankingScore = bestScore;
    rankingScore(accepted) = -inf;
    rankingScore(used) = -inf;
    [~, candidate] = max(rankingScore);
    if ~isfinite(rankingScore(candidate)) && rankingScore(candidate) < 0
        break
    end
    nextIndex = candidate;
end

anchorIndices = anchorIndices(1:anchorCount);
anchors = anchors(1:anchorCount);

fullState = zeros(2 * busCount, sampleCount);
firstAnchor = anchors{1};
fullState(slack, :) = real(firstAnchor.slackVoltage);
fullState(busCount + slack, :) = imag(firstAnchor.slackVoltage);
fullState(pq, :) = bestState(1:numel(pq), :);
fullState(busCount + pq, :) = bestState(numel(pq) + (1:numel(pq)), :);

atlasResult = struct();
atlasResult.caseName = caseName;
atlasResult.state = bestState;
atlasResult.fullState = fullState;
atlasResult.pq = pq;
atlasResult.slack = slack;
atlasResult.anchorCount = anchorCount;
atlasResult.anchorIndices = anchorIndices;
atlasResult.anchors = anchors;
atlasResult.assignedAnchor = bestAnchor;
atlasResult.stateResidualEstimate = bestScore;
atlasResult.accepted = accepted;
atlasResult.powerResidualInfinity = bestInfinity;
atlasResult.correctionNorm = bestCorrection;
atlasResult.correctionCount = correctionCount;
atlasResult.totalCorrectionCount = totalCorrectionCount;
atlasResult.backtrackCount = backtrackCount;
atlasResult.predictorState = predictorState;
atlasResult.predictorScore = predictorScore;
atlasResult.powerResidualNorm = bestResidual;
atlasResult.localStateBound = bestBound;
atlasResult.maximumStateResidualEstimate = max(bestScore);
atlasResult.maximumPowerResidualNorm = max(bestResidual);
atlasResult.toleranceSatisfiedCount = sum(accepted);
atlasResult.unresolvedCount = sum(~isfinite(bestScore));
atlasResult.sampleCount = sampleCount;
atlasResult.timeSec = toc(tStart);
atlasResult.status = "computed";
end
