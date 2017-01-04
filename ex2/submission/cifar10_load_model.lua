require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'xlua'
local c = require 'trepl.colorize'
---- ### Classification criterion
criterion = nn.CrossEntropyCriterion():cuda()
---	 ### predefined constants
require 'optim'
batchSize = 128

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
local trsize = 50000
local tesize = 10000
-- load the model
model = torch.load('model.net')

----------------------################ code from provider ############-------------------------------
-----------------------------------------------------------------------------------------------------
-- load dataset

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')

local trainData = {
	data = torch.Tensor(50000, 3072),
	labels = torch.Tensor(50000),
	size = function() return trsize end
}
trainData.data = trainset.data:double()
trainData.labels = trainset.label:double():add(1)
--print (c.blue 'after train')
--print(c.blue ''.. trainData.labels)

local testData = {
	data = testset.data:double(),
	labels = testset.label:double():add(1),
	size = function() return tesize end
}

-- preprocess/normalize train/test sets

-- preprocess trainSet
local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1,trainData:size() do
	xlua.progress(i, trainData:size())
	-- rgb -> yuv
	local rgb = trainData.data[i]
	local yuv = image.rgb2yuv(rgb)
	-- normalize y locally:
	yuv[1] = normalization(yuv[{{1}}])
	trainData.data[i] = yuv
end
-- normalize u globally:
local mean_u = trainData.data:select(2,2):mean()
local std_u = trainData.data:select(2,2):std()
trainData.data:select(2,2):add(-mean_u)
trainData.data:select(2,2):div(std_u)
-- normalize v globally:
local mean_v = trainData.data:select(2,3):mean()
local std_v = trainData.data:select(2,3):std()
trainData.data:select(2,3):add(-mean_v)
trainData.data:select(2,3):div(std_v)

trainData.mean_u = mean_u
trainData.std_u = std_u
trainData.mean_v = mean_v
trainData.std_v = std_v

-- preprocess testSet
for i = 1,testData:size() do
	xlua.progress(i, testData:size())
	--print (c.blue 'before forwardnet')
	-- rgb -> yuv
	local rgb = testData.data[i]
	local yuv = image.rgb2yuv(rgb)
	-- normalize y locally:
	yuv[{1}] = normalization(yuv[{{1}}])
	testData.data[i] = yuv
end
-- normalize u globally:
testData.data:select(2,2):add(-mean_u)
testData.data:select(2,2):div(std_u)
-- normalize v globally:
testData.data:select(2,3):add(-mean_v)
testData.data:select(2,3):div(std_v)
----------------------################ end of data preprocess ############-------------------------------
----------------------------------------------------------------------------------------------------
local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float()
local testData_1 = testset.data:float()
local testLabels_1 = testset.label:float()


function test()
	confusion = optim.ConfusionMatrix(10)
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 125
  for i=1,testData.data:size(1),bs do
    local outputs = model:forward(testData.data:narrow(1,i,bs):cuda())
    confusion:batchAdd(outputs, testData.labels:narrow(1,i,bs):cuda())
  end

  confusion:updateValids()
  -- print('Test accuracy:', confusion.totalValid * 100..'%')
  -- print('Test accuracy:',(1- confusion.totalValid) * 100..'%')
  return (1 - confusion.totalValid)
end
-- run on test set
test()






