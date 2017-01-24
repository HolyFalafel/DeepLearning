require 'torch'
require 'nn'
require 'optim'
require 'eladtools'
require 'recurrent'
require 'utils.textDataProvider'
-------------------------------------------------------

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Testing recurrent networks on word-level text dataset - Penn Treebank')
cmd:text()
cmd:text('==>Options')
-- danny
cmd:text('===>Model And Training Regime')
cmd:option('-seqLength',          50,                          'number of timesteps to unroll for')
cmd:option('-batchSize',          50,                          'batch size')
cmd:text('===>Platform Optimization')
cmd:option('-type',               'cuda',                      'float or cuda')
cmd:option('-devid',              1,                           'device ID (if using CUDA)')
cmd:option('-seed',               123,                         'torch manual random number generator seed')
cmd:option('-nGPU',               1,                           'num of gpu devices used')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '/home/dannysem@st.technion.ac.il/DeepLearning/ex3/Results/MonJan2308:27:022017/Net_40.t7',                          'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''),      'save directory')
cmd:option('-optState',           false,                       'Save optimization state every epoch')
cmd:option('-checkpoint',         0,                           'Save a weight check point every n samples. 0 for off')




opt = cmd:parse(arg or {})
--opt.save = paths.concat('./Results', opt.save)
--torch.setnumthreads(opt.threads)
--torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')
--sequential criterion
local criterion = nn.CrossEntropyCriterion()--ClassNLLCriterion()
local seqCriterion = nn.TemporalCriterion(criterion)
local TensorType = 'torch.CudaTensor'
criterion = criterion:cuda()
if opt.type =='cuda' then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.devid)
    cutorch.manualSeed(opt.seed)
end
----------------------------------------------------------------------
local trainWordVec, testWordVec, valWordVec, decoder, decoder_, vocab

trainWordVec, vocab, decoder = loadTextFileWords('./data/ptb.train.txt')
testWordVec, vocab, decoder_ = loadTextFileWords('./data/ptb.test.txt', vocab)
assert(#decoder == #decoder_) --no new words
valWordVec, vocab, decoder_ = loadTextFileWords('./data/ptb.valid.txt', vocab)
assert(#decoder == #decoder_) --no new words
data = {
  trainingData = trainWordVec,
  testData = testWordVec,
  validationData = valWordVec,
  vocabSize = #decoder,
  decoder = decoder,
  vocab = vocab,
  decode = decodeFunc(vocab, 'word'),
  encode = encodeFunc(vocab, 'word')
}
local vocabSize = #decoder

local embedder
local recurrent
local classifier
----------------------------------------------------------------------
-- danny add function
local function reshapeData(wordVec, seqLength, batchSize)
    local offset = offset or 0
    local length = wordVec:nElement()
    local numBatches = torch.floor(length / (batchSize * seqLength))

    local batchWordVec = wordVec.new():resize(numBatches, batchSize, seqLength)
    local endWords = wordVec.new():resize(numBatches, batchSize, 1)

    local endIdxs = torch.LongTensor()
    for i=1, batchSize do
        local startPos = torch.round((i - 1) * length / batchSize ) + 1
        local sliceLength = seqLength * numBatches
        local endPos = startPos + sliceLength - 1

        batchWordVec:select(2,i):copy(wordVec:narrow(1, startPos, sliceLength))
    endIdxs:range(startPos + seqLength, endPos + 1, seqLength)
endWords:select(2,i):copy(wordVec:index(1, endIdxs))
  end
  return batchWordVec, endWords
end
----------------------------------------------------------------------
local function ForwardSeq(dataVec, train, model)

    local data, labels = reshapeData(dataVec, opt.seqLength, opt.batchSize )
    local sizeData = data:size(1)
    local numSamples = 0
    local lossVal = 0
    local currLoss = 0
    local x = torch.Tensor(opt.batchSize, opt.seqLength):type(TensorType)
    local yt = torch.Tensor(opt.batchSize, opt.seqLength):type(TensorType)

    -- input is a sequence
    model:sequence()
    model:forget()

    for b=1, sizeData do
        if b==1 or opt.shuffle then --no dependancy between consecutive batches
            model:zeroState()
        end
        x:copy(data[b])
        yt:narrow(2,1,opt.seqLength-1):copy(x:narrow(2,2,opt.seqLength-1))
        yt:select(2, opt.seqLength):copy(labels[b])

        if train then
            if opt.nGPU > 1 then
                model:syncParameters()
            end
            y, currLoss = optimizer:optimize(x, yt)
        else
            y = model:forward(x)
            currLoss = seqCriterion:forward(y,yt)
        end
        lossVal = currLoss / opt.seqLength + lossVal
        numSamples = numSamples + x:size(1)
        xlua.progress(numSamples, sizeData*opt.batchSize)
    end

    collectgarbage()
    xlua.progress(numSamples, sizeData)
    return lossVal / sizeData
end

local function ForwardSingle(dataVec)
    local sizeData = dataVec:nElement()
    local numSamples = 0
    local lossVal = 0
    local currLoss = 0

    -- input is from a single time step
    model:single()

    local x = torch.Tensor(1,1):type(TensorType)
    local y
    for i=1, sizeData-1 do
        x:fill(dataVec[i])
        y = recurrent:forward(embedder:forward(x):select(2,1))
        currLoss = criterion:forward(y, dataVec[i+1])
        lossVal = currLoss + lossVal
        if (i % 100 == 0) then
            xlua.progress(i, sizeData)
        end
    end

    collectgarbage()
    return(lossVal/sizeData)
end

-----------------------------------------------------------------------------------------
local function evaluate(dataVec, model)
    model:evaluate()
    return ForwardSeq(dataVec, false, model)
end

local function sample(str, num, space, model, embedder, recurrent, classifier, temperature)
    local num = num or 50
    local temperature = temperature or 1
    local function smp(preds)
        if temperature == 0 then
            local _, num = preds:max(2)
            return num
        else
            preds:div(temperature) -- scale by temperature
            local probs = preds:squeeze()
            probs:div(probs:sum()) -- renormalize so probs sum to one
            local num = torch.multinomial(probs:float(), 1):typeAs(preds)
            return num
        end
    end


    recurrent:evaluate()
    recurrent:single()

    local sampleModel = nn.Sequential():add(embedder):add(recurrent):add(classifier):add(nn.SoftMax():type(TensorType))

	-- print(sampleModel)
    local pred, predText, embedded
    if str then
        local encoded = data.encode(str)
		print(encoded) 
		print(encoded:size())
        for i=1, encoded:nElement() do
            -- pred = model:forward(encoded:narrow(1,i,1))
			-- print(encoded:narrow(1,i,1))
			pred = sampleModel:forward(encoded:narrow(1,i,1))
			-- pred = sampleModel:forward(1)
        end
        wordNum = smp(pred)

        predText = str .. '... ' .. decoder[wordNum:squeeze()]
    else
        wordNum = torch.Tensor(1):random(vocabSize):type(TensorType)
        predText = ''
    end

    for i=1, num do
        pred = model:forward(wordNum)
        wordNum = smp(pred)
        if space then
            predText = predText .. ' ' .. decoder[wordNum:squeeze()]
        else
            predText = predText .. decoder[wordNum:squeeze()]
        end
    end
    return predText
end

-----------------------------------------------------------------------------

----------------Testing a loaded model
if paths.filep(opt.load) then
    -- model = torch.load(opt.load)
    modelConfig = torch.load(opt.load)
	-- print(modelConfig)
end
    print('==>Loaded Net from: ' .. opt.load)
	
	-- local trainingConfig = require 'utils.trainRecurrent'
	-- local evaluate = trainingConfig.evaluate
	-- local sample = trainingConfig.sample
	-- danny load model
	 embedder = modelConfig.embedder
	 recurrent = modelConfig.recurrent
	 classifier = modelConfig.classifier
	-- local trainRegime = modelConfig.regime
	local stateSize = modelConfig.stateSize
	 vocab = modelConfig.vocab
	 decoder = modelConfig.decoder
	 model = modelConfig.model

	-- -- Model + Loss:
	-- local model = nn.Sequential()
	-- model:add(embedder)
	-- model:add(recurrent)
	-- model:add(nn.TemporalModule(classifier))
	-- print '\n==> Network'
	-- print(model)

--local trainingConfig = require 'utils.trainRecurrent'
--local evaluate = trainingConfig.evaluate
--local sample = trainingConfig.sample

--print('\nEpoch ' .. epoch ..'\n')

local LossTest = evaluate(data.testData, model)
print('\nTest Perplexity: ' .. torch.exp(LossTest))

print('\nSampled Text:\n' .. sample('Buy low, sell high is the', 50, true, model, embedder, recurrent, classifier))



