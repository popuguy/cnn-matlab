% setup testing
pl1 = PoolLayer(2,2);
pl2 = PoolLayer(3,3);

testCellArray1 = cell(1,1);
testCellArray1{1} = ones(6);
testCellArray2 = cell(1,3);
testCellArray2{1} = ones(6);
testCellArray2{2} = ones(6);
testCellArray2{3} = ones(6);

%% Test 1: single channel input
poolExpectedResult = cell(1,1);
poolExpectedResult{1} = ones(3) * 4;
poolResult = pl1.forward(testCellArray1);
assert(isequal(poolResult, poolExpectedResult), 'One channel cell array not pooling properly with 2x2 pool');

%% Test 2: multi-channel input
poolExpectedResult = cell(1,3);
poolExpectedResult{1} = ones(2) * 9;
poolExpectedResult{2} = ones(2) * 9;
poolExpectedResult{3} = ones(2) * 9;
poolResult = pl2.forward(testCellArray2);
assert(isequal(poolResult, poolExpectedResult), 'Multi-channel cell array not pooling properly with 3x3 pool');