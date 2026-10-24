function reducedPower = evaluate_reduced_power(reducedState, anchor)
%EVALUATE_REDUCED_POWER Evaluate the exact rectangular AC power map in batch.

arguments
    reducedState (:, :) double
    anchor (1, 1) struct
end

npq = numel(anchor.pq);
assert(size(reducedState, 1) == 2 * npq, ...
    "BTMM:powerflow:StateDimension", ...
    "reducedState must have 2*npq rows.");

sampleCount = size(reducedState, 2);
voltage = repmat(anchor.voltage, 1, sampleCount);
voltage(anchor.pq, :) = complex( ...
    reducedState(1:npq, :), reducedState(npq + (1:npq), :));
current = anchor.Ybus * voltage;
injection = voltage .* conj(current);
reducedPower = [real(injection(anchor.pq, :)); imag(injection(anchor.pq, :))];
end
