function result = refine_fixed_jacobian(state, target, anchor, options)
%REFINE_FIXED_JACOBIAN Batch modified Newton updates; see derivation Section 1.
arguments
    state (:,:) double
    target (:,:) double
    anchor (1,1) struct
    options.MaximumCorrections (1,1) double {mustBeInteger,mustBeNonnegative} = 10
    options.StateTolerance (1,1) double {mustBePositive} = 1e-7
    options.PowerTolerance (1,1) double {mustBePositive} = 1e-8
    options.VoltageMin (1,1) double = .8
    options.VoltageMax (1,1) double = 1.2
end
assert(isequal(size(state),size(target)));
powerMap = @(x) BTMM_core.powerflow.evaluate_reduced_power(x,anchor);
residual = target-powerMap(state);
count = zeros(1,size(state,2));
stalled = false(size(count));
backtracks = zeros(size(count));
maximumStepReductions = 8;
for iteration = 1:options.MaximumCorrections
    [accepted,inside] = check_state(state,residual,anchor,options);
    active = find(~accepted & inside & ~stalled & all(isfinite(residual),1));
    if isempty(active), break; end
    oldNorm = vecnorm(residual(:,active),2,1);
    delta = anchor.inverseMatrix*residual(:,active);
    pending = 1:numel(active);
    step = ones(1,numel(active));
    moved = false(1,numel(active));
    for reduction = 0:maximumStepReductions
        trial = state(:,active(pending))+delta(:,pending).*step(pending);
        trialResidual = target(:,active(pending))-powerMap(trial);
        [~,trialInside] = check_state(trial,trialResidual,anchor,options);
        decreasing = trialInside & all(isfinite(trialResidual),1) & ...
            vecnorm(trialResidual,2,1)<oldNorm(pending);
        chosen = pending(decreasing);
        state(:,active(chosen)) = trial(:,decreasing);
        residual(:,active(chosen)) = trialResidual(:,decreasing);
        count(active(chosen)) = count(active(chosen))+1;
        backtracks(active(chosen)) = backtracks(active(chosen))+reduction;
        moved(chosen) = true;
        pending = pending(~decreasing);
        if isempty(pending), break; end
        step(pending) = step(pending)/2;
    end
    stalled(active(~moved)) = true;
end
[accepted,inside] = check_state(state,residual,anchor,options);
result = struct('state',state,'residual',residual,'accepted',accepted, ...
    'voltageInside',inside,'correctionCount',count,'backtrackCount',backtracks, ...
    'stalled',stalled,'stateResidualEstimate',anchor.inverseNorm*vecnorm(residual,2,1), ...
    'correctionNorm',vecnorm(anchor.inverseMatrix*residual,2,1), ...
    'powerResidualInfinity',max(abs(residual),[],1));
end

function [accepted,inside] = check_state(state,residual,anchor,options)
n = numel(anchor.pq);
magnitude = hypot(state(1:n,:),state(n+(1:n),:));
inside = all(isfinite(state),1) & all(magnitude>=options.VoltageMin & ...
    magnitude<=options.VoltageMax,1);
accepted = inside & all(isfinite(residual),1) & ...
    anchor.inverseNorm*vecnorm(residual,2,1)<=options.StateTolerance & ...
    max(abs(residual),[],1)<=options.PowerTolerance;
end
