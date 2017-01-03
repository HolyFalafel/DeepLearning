require 'xlua'
require 'optim'
require 'nn'
require 'image'
require 'xlua'


local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default cudnn)            backend
   --type                     (default cuda)          cuda/float/cl
]]

-- danny override
opt.model = 'our_model'
opt.type = 'cuda'
opt.backend = 'cudnn'
--opt.max_epoch = 2
print(opt)

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end
end

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
model:add(cast(dofile('models/'..opt.model..'.lua')))
model:get(2).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.benchmark=true
   cudnn.convert(model:get(3), cudnn)
end

print(model)

print(c.blue '==>' ..' loading data')

----------------------################ preprocess - code from provider ############-------------------------------
-----------------------------------------------------------------------------------------------------
-- load dataset
local trsize = 50000
local tesize = 10000
local trainset = torch.load('../cifar.torch/cifar10-train.t7')
local testset = torch.load('../cifar.torch/cifar10-test.t7')

local trainData = {
  data = torch.Tensor(50000, 3072),
  labels = torch.Tensor(50000),
  size = function() return trsize end
}
trainData.data = trainset.data:double()
trainData.labels = trainset.label:double():add(1)

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

trainData.data = trainData.data:float()
testData.data = testData.data:float()

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = cast(nn.CrossEntropyCriterion())


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  -- danny add loss-error
  local lossAcc = 0
  local numBatches = 0
  local targets = cast(torch.FloatTensor(opt.batchSize))
  local indices = torch.randperm(trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    -- danny add loss-error
	numBatches = numBatches + 1
	xlua.progress(t, #indices)
    xlua.progress(t, #indices)

    local inputs = trainData.data:index(1,v)
    targets:copy(trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)

	  -- danny add loss-error
	  y = outputs
	  err = f
	  lossAcc = lossAcc + err
	  
      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  -- danny add loss-error
  local avgLoss = lossAcc / numBatches
  local avgError = 1 - confusion.totalValid
  
  confusion:zero()
  epoch = epoch + 1
  
  -- danny add loss-error
  return avgLoss, avgError
end

epochs = opt.max_epoch
trainLoss = torch.Tensor(epochs):zero()
testLoss = torch.Tensor(epochs):zero()
trainError = torch.Tensor(epochs):zero()
testError = torch.Tensor(epochs):zero()

-- ********************* Test Plots *********************

require 'gnuplot'
local range = torch.range(1, epochs)
local filename = paths.concat(opt.save, 'testLoss.png')
gnuplot.pngfigure(filename)
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()

filename = paths.concat(opt.save, 'testError.png')
gnuplot.pngfigure(filename)
gnuplot.plot({'trainError',trainError},{'testError',testError})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Error')
gnuplot.plotflush()

function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  
  -- danny add loss
  local lossAcc = 0
  local numBatches = 0
  local bs = 125
  for i=1,testData.data:size(1),bs do
    numBatches = numBatches + 1
    local outputs = model:forward(testData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, testData.labels:narrow(1,i,bs))
	
	-- danny add loss
	local yt = testData.labels:narrow(1,i,bs)
	local y = outputs
	local err = criterion:forward(y, yt)
	lossAcc = lossAcc + err
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    if paths.filep(opt.save..'/test.log.eps') then
      local base64im
      do
        os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
        os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
        local f = io.open(opt.save..'/test.base64')
        if f then base64im = f:read'*all' end
      end

      local file = io.open(opt.save..'/report.html','w')
      file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <title>%s - %s</title>
      <img src="data:image/png;base64,%s">
      <h4>optimState:</h4>
      <table>
      ]]):format(opt.save,epoch,base64im))
      for k,v in pairs(optimState) do
        if torch.type(v) == 'number' then
          file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
        end
      end
      file:write'</table><pre>\n'
      file:write(tostring(confusion)..'\n')
      file:write(tostring(model)..'\n')
      file:write'</pre></body></html>'
      file:close()
    end
	
  end

  -- save model and graphs every 50 epochs
  if epoch % 50 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3):clearState())
	
	-- ********************* Plots *********************

	require 'gnuplot'
	local range = torch.range(1, epochs)
	local filename = paths.concat(opt.save, 'testLoss.png')
	gnuplot.pngfigure(filename)
	gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Loss')
	gnuplot.plotflush()

	filename = paths.concat(opt.save, 'testError.png')
	gnuplot.pngfigure(filename)
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
  end

  -- danny add loss-error
  local avgLoss = lossAcc / numBatches
  local avgError = 1 - confusion.totalValid
  
  confusion:zero()
  
  -- danny add loss-error
  return avgLoss, avgError
end


bestEpoch = 0
lowestError = 1

for e=1,opt.max_epoch do
  -- danny add loss-error
  trainLoss[e], trainError[e] = train()
  testLoss[e], testError[e] = test()
  print("trainLoss")
  print(trainLoss[e])
  print("trainError")
  print(trainError[e])
  print("testLoss")
  print(testLoss[e])
  print("testError")
  print(testError[e])
  
  if lowestError > testError[e] then
	bestEpoch = e
	lowestError = testError[e]
  end

end

print('best Epoch:')
print(bestEpoch)
print('lowestError:')
print(lowestError)

-- ********************* Plots *********************

require 'gnuplot'
local range = torch.range(1, epochs)
local filename = paths.concat(opt.save, 'testLoss.png')
gnuplot.pngfigure(filename)
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()

filename = paths.concat(opt.save, 'testError.png')
gnuplot.pngfigure(filename)
gnuplot.plot({'trainError',trainError},{'testError',testError})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Error')
gnuplot.plotflush()
