local mnist = require 'mnist';
local mnist = require 'mnist';
require 'cifar10_classification_ex2_augmentation.lua'
require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'

---- ### Classification criterion
criterion = nn.ClassNLLCriterion():cuda()
---	 ### predefined constants
require 'optim'
batchSize = 128

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local function forwardNet(data, labels)
	timer = torch.Timer()

    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
	local numBatches = 0
	
	model:evaluate() -- turn off drop-out

    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        --local err = criterion:forward(y, yt)
        --lossAcc = lossAcc + err
		print('#x')
		print(#x)
		print('#y')
		print(#y)
		print('#yt')
		print(#yt)
        confusion:batchAdd(y,yt)
    end
    
    confusion:updateValids()
    local avgError = 1 - confusion.totalValid

	return avgError
end

function load_model()
	-- load the model
	model = torch.load('network.model')
	
	local trainset = torch.load('cifar.torch/cifar10-train.t7')
	local testset = torch.load('cifar.torch/cifar10-test.t7')

	local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
	local trainLabels = trainset.label:float():add(1)
	local testData = testset.data:float()
	local testLabels = testset.label:float():add(1)
	
	--We'll start by normalizing our data
	local mean = {}  -- store the mean, to normalize the test set in the future
	local stdv  = {} -- store the standard-deviation for the future
	for i=1,3 do -- over each image channel
		mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
		-- print('Channel ' .. i .. ', Mean: ' .. mean[i])
		trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
		
		stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
		-- print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
		trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
	end

	-- Normalize test set using same values

	for i=1,3 do -- over each image channel
		testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
		testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
	end
	
	print("#data")
	print(#testData)
	
	print("#labels")
	print(#testLabels)
	-- run on test set
	testError = forwardNet(testData, testLabels)

	return testError
end










