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
----------------------------------------------------------------------
----------------------------------------------------------------------
local function ForwardSeq(dataVec, train)

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
local function evaluate(dataVec)
    model:evaluate()
    return ForwardSeq(dataVec, false)
end

local function sample(str, num, space, temperature)
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


    --recurrent:evaluate()
    --recurrent:single()

    --local sampleModel = nn.Sequential():add(embedder):add(recurrent):add(classifier):add(nn.SoftMax():type(TensorType))

    local pred, predText, embedded
    if str then
        local encoded = data.encode(str)
		print(#encoded) 
		print(encoded:size())
        for i=1, encoded:nElement() do
            pred = model:forward(encoded:narrow(1,i,1))
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
	
end
    print('==>Loaded Net from: ' .. opt.load)
	-- print '\n==> Network'
	-- print(model)

--local trainingConfig = require 'utils.trainRecurrent'
--local evaluate = trainingConfig.evaluate
--local sample = trainingConfig.sample

--print('\nEpoch ' .. epoch ..'\n')

local LossTest = evaluate(data.testData)

print('\nSampled Text:\n' .. sample('Buy low, sell high is the', 50, true))

print('\nTest Perplexity: ' .. torch.exp(LossTest))


