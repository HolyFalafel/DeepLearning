
--[[ 
make sure to preform: 
luarocks install mnist
luarocks install image
]]
require 'image'
local mnist = require 'mnist';
--local optim = 
require 'optim'
local trainData = mnist.traindataset();
local testData = mnist.testdataset();

trainData = trainData.data:float(); --turn data to float (originaly byte)
testData = testData.data:float();

trainData:add(-127):div(128); -- centre data around 0 !!!!!!!!!!!!!!!!!
testData:add(-127):div(128);
--[[

print(trainData:size())
print(testData:size())

local i = torch.random(1,trainData:size(1))
image.save('train'..i..'.PNG', trainData:select(1,i))
]]

-- ********************* Define the model *********************
--make sure to preform: luarocks install; nn  luarocks install cunn

require 'nn'
require 'cunn'

local hSize = 32
model = nn.Sequential()
model:add(nn.View(28 * 28)) --reshapes the image into a vector without copy
model:add(nn.Linear(28 * 28, hSize))
model:add(nn.ReLU())
model:add(nn.Linear(hSize, 16))
model:add(nn.Tanh())
model:add(nn.Linear(16, 3))
model:add(nn.Sigmoid())
model:add(nn.Linear(3, 16))
model:add(nn.Sigmoid())
model:add(nn.Linear(16, hSize))
model:add(nn.Tanh())
model:add(nn.Linear(hSize, 28 * 28))
model:add(nn.Tanh())
model:add(nn.View(28, 28))
model:cuda()  -- ship to GPU

-- check that we can propagate forward without errors
-- the output should be 28X28 Tensor
print(#model:forward(trainData:select(1,1):cuda()))
print(model)


-- ********************* criterion *********************

local criterion = nn.MSECriterion():cuda()
local w, dE_dw = model:getParameters()

local linears = model:findModules('nn.Linear')
print (linears)
print('Number total of parameters:', w:nElement())


-- ********************* Training *********************

local lr = 0.1 -- learning rate used for sgd
local batchSize = 16 -- when bigger - more use of gpu (16 operations in one time unit)
local epochs = 25
local printEval = true
local trainLoss = torch.Tensor(epochs)
local testLoss = torch.Tensor(epochs)

local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())

local timer = torch.Timer()

for e = 1, epochs do
    local lossAcc = 0 -- loss...
    local numBatches = 0 -- num of batches..
    
    for i = 1, trainData:size(1), batchSize do
        numBatches = numBatches + 1
        local x = trainData:narrow(1, i, batchSize):cuda() -- Looking at 16 pictures... average of 16 pictures...
        local y = model:forward(x) -- predicting...
        lossAcc = lossAcc + criterion:forward(y, x) -- criterion - using MSE to calculate error...

        model:zeroGradParameters() --zero grads (dE_dw..)
        local dE_dy = criterion:backward(y,x) -- criterion deriviation...
        model:backward(x, dE_dy) -- backpropagation -- deriviation of loss ||y-y^||^2...
        -- dE_dw now has the gradient of all the 16 examples...
		--|linear|->|ReLU|->|output|
		
        w:add(-lr, dE_dw) -- naive sgd: w' = w - lr * dE_dw
    end
    
	-- e - epoch (see loss of each epoch)
    trainLoss[e] = lossAcc / numBatches
    
    --Evaluate on test set
    lossAcc = 0
    numBatches = 0
    
    for i = 1, testData:size(1), batchSize do
        numBatches = numBatches + 1
        local x = testData:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        lossAcc = lossAcc + criterion:forward(y, x)
    end
    
    testLoss[e] = lossAcc / numBatches
    
    if printEval then
        print('epoch number: ' .. e .. '\n')
        print('Training Loss:', trainLoss[e])
        print('Test Loss:', testLoss[e])
    
        --view reconstructed images
        x = testData:narrow(1, 1, 36):cuda()
        y = model:forward(x)
		for t = 1, 5 do
			--image.save('epoch'..e..'_test' ..t..'.PNG', y:select(1,t))
		end 
    end
  
end

print(timer:time().real .. ' seconds')


-- ********************* Plots *********************

require 'gnuplot'

local range = torch.range(1, epochs)

gnuplot.pngfigure('test.png')
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()







