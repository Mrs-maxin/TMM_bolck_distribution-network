function mapped = map_power_samples_quadratic( ...
    anchorState, anchorPower, targetPower, inverseAction, powerMap)
%MAP_POWER_SAMPLES_QUADRATIC Apply the quadratic local inverse in a batch.
%   The calculation uses two exact power-map evaluations and three actions
%   of the anchor inverse Jacobian. POWERMAP and INVERSEACTION must accept
%   matrices whose columns are samples. See 01_IDEA/derivation.md, Section 4.

arguments
    anchorState (:, 1) double
    anchorPower (:, 1) double
    targetPower (:, :) double
    inverseAction (1, 1) function_handle
    powerMap (1, 1) function_handle
end

stateCount = numel(anchorState);
assert(numel(anchorPower) == stateCount, ...
    "BTMM:inverse:AnchorDimensionMismatch", ...
    "anchorState and anchorPower must have the same reduced dimension.");
assert(size(targetPower, 1) == stateCount, ...
    "BTMM:inverse:TargetDimensionMismatch", ...
    "targetPower must have one row per reduced PQ state.");

powerIncrement = targetPower - anchorPower;
firstIncrement = inverseAction(powerIncrement);
assert(isequal(size(firstIncrement), size(targetPower)), ...
    "BTMM:inverse:InverseActionShape", ...
    "inverseAction must preserve the reduced sample-matrix shape.");

firstState = anchorState + firstIncrement;
firstMismatch = powerMap(firstState) - targetPower;
secondIncrement = -inverseAction(firstMismatch);
predictorState = firstState + secondIncrement;

predictorResidual = targetPower - powerMap(predictorState);
correctedState = predictorState + inverseAction(predictorResidual);
correctedResidual = targetPower - powerMap(correctedState);

mapped = struct();
mapped.state = correctedState;
mapped.predictorState = predictorState;
mapped.firstIncrement = firstIncrement;
mapped.secondIncrement = secondIncrement;
mapped.predictorResidual = predictorResidual;
mapped.correctedResidual = correctedResidual;
mapped.sampleCount = size(targetPower, 2);
mapped.status = "computed";
end
