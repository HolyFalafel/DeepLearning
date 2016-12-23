require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

function saveTensorAsGrid(tensor,fileName) 
	local padding = 2
	local grid = image.toDisplayTensor(tensor, padding)
	image.save(fileName,grid)
end

local dataset = torch.load('flowers.t7')
local classes = torch.range(1,17):totable() -- 17 classes
local labels = torch.range(1,17):view(17,1):expand(17,80)

print(dataset:size()) -- 17x80x3x128x128
saveTensorAsGrid(dataset:select(2,1),'flowers.jpg')

----------------------------------------------------------------
-- split to train and test
----------------------------------------------------------------
--[[
function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

-- contiguous = no memory copy if possibale 
local shuffledData, shuffledLabels = shuffle(dataset:view(-1,3,128,128), labels:contiguous():view(-1))

local trainSize = 0.9 * shuffledData:size(1)
-- unpack receives an array and returns as results all 
-- elements from the array, starting from index 1:
local trainData, testData = unpack(shuffledData:split(trainSize, 1))
local trainLabels, testLabels = unpack(shuffledLabels:split(trainSize, 1))

print(trainData:size())
]]
-----------------------------------------------------------------
-- Data preprocess
-----------------------------------------------------------------

--[[
trainData = trainData:float() 
trainLabels = trainLabels:float()

local mean, std = trainData:mean(), trainData:std()
print(mean, std)
trainData:add(-mean):div(std)
    
testData = testData:float()
testLabels = testLabels:float()
testData:add(-mean):div(std)
]]

-----------------------------------------------------------------
-- Using preTrained network
-----------------------------------------------------------------
--[[
local googLeNet = torch.load('GoogLeNet_v2.t7')
--print(tostring(googLeNet))
]]
-----------------------------------------------------------------
-- Chopping the network
-----------------------------------------------------------------
--[[
local model = nn.Sequential()

for i=1,10 do
    local layer = googLeNet:get(i):clone()
    layer.parameters = function() return {} end --disable parameters
	--remove accGradParamters (make sure those layers will not be trained)
    layer.accGradParamters = nil 
    model:add(layer)
end

model:cuda()
local y = model:forward(torch.rand(1,3,128,128):cuda())
print(y:size())
]]
-----------------------------------------------------------------
-- Adding classifier layers  !!!!!
-----------------------------------------------------------------
--[[
model:add(cudnn.SpatialConvolution(320, 16, 3,3)) -- 16x14x14
model:add(cudnn.ReLU())
model:add(cudnn.SpatialMaxPooling(4,4))  -- 16x3x3
model:add(nn.View(16*3*3):setNumInputDims(3)) -- 1x144
model:add(nn.Dropout(0.5))
model:add(nn.Linear(16*3*3, #classes))
model:add(nn.LogSoftMax())

model:cuda()
criterion = nn.ClassNLLCriterion():cuda()
w, dE_dw = model:getParameters()
print('#Parameters = ', #w)
]]
-----------------------------------------------------------------
-- Training the network
-----------------------------------------------------------------
--[[
require 'optim'

batchSize = 16
optimState = {}


function forwardNet(data,labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate()
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
            optim.adam(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

function plotError(trainError, testError, title)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end

---------------------------------------------------------------------

epochs = 25
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)



for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    
    if e % 2 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
    end
end

plotError(trainError, testError, 'Classification Error')

]]