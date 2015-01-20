-- require('mobdebug').start()
require 'nn'

mnist = require 'mnist'
trainset = mnist.traindataset()
testset = mnist.testdataset()

train_size = trainset.size
test_size = testset.size
function trainset:size() return train_size end
function testset:size() return test_size end

function lookup(table, index)
  if index >= 1 and index <= table.size() then
    return {
      table.data[index]:double(),
      table.label[index] == 0 and 10 or table.label[index]
    }
  end
end

setmetatable(trainset, {__index = lookup})
setmetatable(testset, {__index = lookup})

-- trainset.data:float()
-- testset.data:float()

mlp = nn.Sequential()
mlp:add(nn.Reshape(784))
mlp:add(nn.Linear(784, 500))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(500, 500))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(500, 500))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(500, 10))
mlp:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01

-- dataset = {}
-- function dataset:size() return 100 end
-- for i=1, dataset:size() do
--   local input = torch.Tensor()
--   local output = torch.Tensor({1,6,7,3,2,9,2,4,10,-5})
--   dataset[i] = {input, output}
-- end

-- trainer:train(trainset)
