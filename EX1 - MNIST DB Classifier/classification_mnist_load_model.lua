local mnist = require 'mnist';
local mnist = require 'mnist';
require 'nn'
require 'cunn'

---- ### Classification criterion
criterion = nn.ClassNLLCriterion():cuda()
---	 ### predefined constants
require 'optim'
batchSize = 128

local function forwardNet(data, labels)
	timer = torch.Timer()

    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local lossAcc = 0
	local numBatches = 0

    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
    end
    
    confusion:updateValids()
    local avgError = 1 - confusion.totalValid

	return avgError
end

function load_model()
	-- load the model
	model = torch.load('network.model')
	
	local trainData = mnist.traindataset().data:float();
	local trainLabels = mnist.traindataset().label:add(1);
	testLabels = mnist.testdataset().label:add(1);
	testData = mnist.testdataset().data:float();
	--We'll start by normalizing our data
	local mean = trainData:mean()
	local std = trainData:std()
	testData:add(-mean):div(std);
	
	-- run on test set
	testError = forwardNet(testData, testLabels)

	return testError
end










