require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
--require './dataAugmentation.lua'

function saveTensorAsGrid(tensor,fileName) 
	local padding = 1
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)

--print(trainData:size())

--saveTensorAsGrid(trainData:narrow(1,100,36),'train_100-136.jpg') -- display the 100-136 images in dataset
--print(classes[trainLabels[100]]) -- display the 100-th image class


--  ****************************************************************
--  Full Example - Training a ConvNet on Cifar10
--  ****************************************************************

-- Load and normalize data:

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end



--  ****************************************************************
--  Define our neural network
--  ****************************************************************
local model = nn.Sequential()

local function Block(...)
  local arg = {...}
  model:add(nn.SpatialConvolution(...))
  model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
  model:add(nn.ReLU(true))
  return model
end

--danny
--model:add(nn.BatchFlip():float())
Block(3,128,5,5,1,1,2,2)
Block(128,32,1,1)
Block(32,16,1,1)
model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
model:add(nn.Dropout())
Block(16,32,5,5,1,1,2,2)
Block(32,64,1,1)
--Block(64,32,1,1)
--Block(192,192,1,1)
model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.Dropout())
----Block(32,32,3,3,1,1,1,1)
Block(64,32,3,3,1,1,1,1)
--Block(192,192,1,1)
Block(32,10,1,1)
model:add(nn.SpatialAveragePooling(8,8,1,1):ceil())
model:add(nn.View(10))
model:add(nn.LogSoftMax()) 

model:cuda()
criterion = nn.ClassNLLCriterion():cuda()


w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

local batchSize = 128
local optimState = {}

function forwardNet(data,labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn off drop-out
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
print("epoch time")
epochs = 250 --50
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

print("after loss calc")

--reset net weights
model:apply(function(l) l:reset() end)

timer = torch.Timer()

print('after timer')

for e = 1, epochs do
    print ('b4 shuffle')
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    print ('after shuffle')
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    print('after train loss')
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    print ('after test loss')

    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
end

plotError(trainError, testError, 'Classification Error')


--  ****************************************************************
--  Network predictions
--  ****************************************************************


model:evaluate()   --turn off dropout

print(classes[testLabels[10] ])
print(testData[10]:size())
saveTensorAsGrid(testData[10],'testImg10.jpg')
local predicted = model:forward(testData[10]:view(1,3,32,32):cuda())
print(predicted:exp()) -- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 


print('saving the model as network.model')
-- save the model
torch.save('network.model', model)

