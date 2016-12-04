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

	-- predicting
	local x = data:cuda()
	local yt = labels:cuda()
	local y = model:forward(x)
	local err = criterion:forward(y, yt)
	lossAcc = lossAcc + err
	confusion:batchAdd(y,yt)
    
    confusion:updateValids()
    local avgError = 1 - confusion.totalValid

	return avgError
end

function load_model()
	-- load the model
	model = torch.load('network.model')
	
	testLabels = mnist.testdataset().label:add(1);
	testData = mnist.testdataset().data:float();
	--We'll start by normalizing our data
	local mean = testData:mean()
	local std = testData:std()
	testData:add(-mean):div(std);
	
	-- run on test set
	testError = forwardNet(testData, testLabels)

	return testError
end










