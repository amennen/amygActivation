toml_file = '/Users/amennen/github/rt-cloud/projects/amygActivation/conf/amygActivation.toml';
dbstop if error;
%toml_file = '/Data1/code/rt-cloud/projects/greenEyes/conf/greenEyes_organized.toml';
runNum = 1;
% if you want it to just be a transfer run w/o neurofeedback, change the
% parameter in the toml file: [display] --> rtData
RealTimeAmygActivation_CLOUD(toml_file, runNum)

% to do:
% - fix text height
% - there's a part after the first round that stops with the rectangle -
% not sure why