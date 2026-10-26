function anchor = build_anchor(caseName, targetP, targetQ, options)
%BUILD_ANCHOR Solve one AC power flow and factor its reduced Jacobian.
%   TARGETP and TARGETQ are full-bus net injections in per unit. The returned
%   anchor acts on the non-slack rectangular state [Ve; Vf] and reduced power
%   vector [P; Q].

arguments
    caseName (1, 1) string
    targetP (:, 1) double
    targetQ (:, 1) double
    options.PowerFlowTolerance (1,1) double {mustBePositive} = 1e-8
end

define_constants;
mpc = loadcase(char(caseName));
n = size(mpc.bus, 1);
assert(numel(targetP) == n && numel(targetQ) == n, ...
    "BTMM:powerflow:InjectionDimension", ...
    "targetP and targetQ must contain one value per bus.");

slack = find(mpc.bus(:, BUS_TYPE) == REF, 1);
assert(isscalar(find(mpc.bus(:,BUS_TYPE)==REF)), 'BTMM:powerflow:SlackBus', ...
    'One fixed slack bus is required.');
pq = setdiff((1:n).', slack, "stable");
assert(all(mpc.bus(pq,BUS_TYPE)==PQ) && ...
    all(mpc.gen(mpc.gen(:,GEN_STATUS)>0,GEN_BUS)==slack), ...
    'BTMM:powerflow:BusTypes','Fixed PQ buses and slack-only generation are required.');

% For the case33 PQ buses, setting demand to minus the requested net
% injection exactly reproduces the sampled operating point. The slack-bus
% injection is determined by the AC balance and is therefore not imposed.
mpc.bus(pq, PD) = -targetP(pq) * mpc.baseMVA;
mpc.bus(pq, QD) = -targetQ(pq) * mpc.baseMVA;

mpopt = mpoption("verbose", 0, "out.all", 0, "pf.tol", options.PowerFlowTolerance);
[solved, success] = runpf(mpc, mpopt);
if ~success
    error("BTMM:powerflow:AnchorFailed", ...
        "AC power flow failed while building a TMM anchor.");
end

vm = double(solved.bus(:, VM));
va = deg2rad(double(solved.bus(:, VA)));
voltage = vm .* exp(1i * va);
Ybus = makeYbus(solved.baseMVA, solved.bus, solved.branch);
injection = voltage .* conj(Ybus * voltage);

[dSdVa, dSdVm] = dSbus_dV(Ybus, voltage);
jacobianPolar = [ ...
    real(dSdVa(pq, pq)), real(dSdVm(pq, pq)); ...
    imag(dSdVa(pq, pq)), imag(dSdVm(pq, pq))];

angle = va(pq);
magnitude = vm(pq);
polarToRectangular = [ ...
    diag(-magnitude .* sin(angle)), diag(cos(angle)); ...
    diag( magnitude .* cos(angle)), diag(sin(angle))];

reducedState = [real(voltage(pq)); imag(voltage(pq))];
reducedPower = [real(injection(pq)); imag(injection(pq))];
inverseMatrix = polarToRectangular * (jacobianPolar \ eye(2 * numel(pq)));

anchor = struct();
anchor.caseName = caseName;
anchor.nBus = n;
anchor.slack = slack;
anchor.pq = pq;
anchor.state = reducedState;
anchor.power = reducedPower;
anchor.voltage = voltage;
anchor.slackVoltage = voltage(slack);
anchor.Ybus = Ybus;
anchor.jacobianPolar = jacobianPolar;
anchor.polarToRectangular = polarToRectangular;
anchor.inverseMatrix = inverseMatrix;
anchor.inverseNorm = norm(inverseMatrix, 2);
anchor.admittanceNorm = norm(full(Ybus), 2);
anchor.reciprocalCondition = rcond(full(jacobianPolar));
anchor.status = "computed";
end
