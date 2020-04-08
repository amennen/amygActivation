function [fileAvail specificFile] = GetSpecificClassOutputFile(classOutputDir,runNum,stationNum,usepython)
runStr = num2str(runNum);
stationStr = num2str(stationNum);
if ~usepython
    specificFile = ['classOutput_r' runStr '_st' stationStr '.mat'];
else
    specificFile = ['classOutput_r' runStr '_st' stationStr '_py.txt'];
end
if exist(fullfile(classOutputDir,specificFile),'file');
    fileAvail = 1;
else
    fileAvail = 0;
end