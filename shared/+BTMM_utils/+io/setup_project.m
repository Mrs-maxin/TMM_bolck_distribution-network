function root = setup_project()
%SETUP_PROJECT Resolve project paths and verify MATLAB dependencies.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
matpowerRoot = getenv('MATPOWER_ROOT');
if ~isempty(matpowerRoot)
    addpath(genpath(matpowerRoot));
end
% Project cases take precedence over same-named MATPOWER example cases.
addpath(root, fullfile(root, 'data', 'cases'));
required = {'runpf','makeYbus','dSbus_dV','fitgmdist','tcdf','trnd'};
for index = 1:numel(required)
    assert(exist(required{index}, 'file') > 0, 'BTMM:Dependency', ...
        'Missing %s. Add MATPOWER and Statistics Toolbox to MATLAB path.', required{index});
end
end
