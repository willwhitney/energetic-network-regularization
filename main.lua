require 'nn'
require 'vis'

norm = 1/2
inputDimension = 4
nHidden = 20

function makeNetwork()
  local mlp = nn.Sequential()
  mlp:add(nn.Linear(inputDimension, nHidden))
  mlp:add(nn.Tanh())
  mlp:add(nn.Linear(nHidden, 2))
  mlp:add(nn.LogSoftMax())
  return mlp
end

criterion = nn.ClassNLLCriterion()

function getDataset(dimension, size)
  dataset = {}
  setmetatable(dataset, {__index = function(table, index)
    local d = torch.rand(dimension) - .5
    local label = d[1] > 0 and 2 or 1
    -- table[index] = {d, label}
    return {d, label}
  end})
  function dataset:size() return size end
  return dataset
end

function getEnergyCost(module)
  -- local energy = 0
  -- for i = 1, module.weights:size()[1] do
      -- local weights = module.weights[i]
    -- energy = energy + torch.tanh(torch.dot(weights, input)) ^ (1/2)
  -- end
  -- energy = energy^2
  -- return energy
  return module.output:norm(norm)
end

function updateParamsMasked(module, learningRate)
  local oldParams, _ = module:parameters()
  paramsMask = {}
  for i, paramTensor in ipairs(oldParams) do
    table.insert(paramsMask, torch.pow(torch.sign(paramTensor), 2))
  end

  module:updateParameters(learningRate)
  local newParams, _ = module:parameters()
  for i, paramTensor in ipairs(newParams) do
    paramTensor:cmul(paramsMask[i])
  end
  local finalParams, _ = module:parameters()
end

function updateAllParamsMasked(network, learningRate)
  local linears = network:findModules('nn.Linear')
  for i, module in ipairs(linears) do
    updateParamsMasked(module, learningRate)
  end
end

function runTestpoint(...)
  local testpoint = torch.zeros(inputDimension)
  testpoint[1] = 0.5
  local networks = table.pack(...)
  for i, network in ipairs(networks) do
    network:forward(testpoint)
    -- print("\nenergy and output of network #"..tostring(i)..":")
    -- print(getEnergyCost(network.modules[2]))
    -- print(network.modules[2].output)
  end
  showOutputDistributions(...)
end

function drink(network, threshold)
  local zeroed = 0
  for i=1, network.modules[1].output:size()[1] do
    if math.abs(network.modules[1].output[i]) < threshold then
      zeroed = zeroed + 1
      network.modules[1].weight[i] = 0
      network.modules[1].bias[i] = 0
    end
  end
  print("Zeroed nodes: "..zeroed.." (of ".. network.modules[1].output:size()[1] .." total)")
  runTestpoint(network)
  -- return zeroed
end

function getRegularizationUpdates(network, input, verbose)
  local lin = network.modules[1]
  local sig = network.modules[2]

  local outer = sig.output:norm(norm)
  local inputSigns = torch.sign(input)
  local outputSigns = torch.sign(sig.output)

  local updates = {}
  for nodeIndex = 1, lin.weight:size()[1] do
    -- local inner = math.tanh(math.abs(sig.output[nodeIndex]))
    local inner = 1

    local weights = lin.weight[nodeIndex]
    local weightContributions = torch.abs(torch.cmul(weights, input))
    local updateSigns = - inputSigns * outputSigns[nodeIndex]

    if verbose then
      print('weights:', weights)
      print('weightContributions:', weightContributions)
      print('inputSigns:', inputSigns)
      print('outputSigns[nodeIndex]:', outputSigns[nodeIndex])
      print('updateSigns:', updateSigns)
    end

    -- update to a weight should be proportional to:
    -- - L1/2 norm of the layer's output
    -- - sqrt of node's output
    -- - contribution of that weight
    -- - the energyReg amount
    -- and point in the direction which will decrease the magnitude of the output
    -- updates[nodeIndex] = torch.cmul(weightContributions, updateSigns) * inner * outer * energyReg
    -- updates[nodeIndex] = updateSigns * inner * outer * energyReg
    updates[nodeIndex] = outer * energyReg
  end
  return updates
end

learningRate = 0.01
energyReg = 0.0001

local batch_size = 10
local num_epochs = 10
local dataset_size = 1000

criterion = nn.ClassNLLCriterion()
dataset = getDataset(inputDimension, dataset_size)

function train(...)
  local arguments = table.pack(...)
  local networks = arguments
  local regularize = false
  if type(arguments[#arguments]) == 'boolean' then
    regularize = arguments[#arguments]
    table.remove(networks, #arguments)
  end

  for networkIndex, network in ipairs(networks) do
    print("Network "..networkIndex)
    print("====================================")
    for epoch = 1,num_epochs do
      local err = 0
      for minibatch = 1, dataset:size() / batch_size do
        for sample = (minibatch - 1) * batch_size + 1, minibatch * batch_size do
          local input = dataset[sample][1]
          local target = dataset[sample][2]
          err = err + criterion:forward(network:forward(input), target)
          network:zeroGradParameters()
          network:backward(input, criterion:backward(network.output, target))
          -- network:updateParameters(learningRate)
          updateAllParamsMasked(network, learningRate)

          if regularize then
            local updates = getRegularizationUpdates(network, input)
            for i = 1, #updates do
              network.modules[1].weight[i]:add(updates[i])
            end
          end
        end
      end
      print(err / dataset:size())
    end
  end
  -- standardize the input for examination
  runTestpoint(table.unpack(networks))
end

function setup()
  mlp = makeNetwork()
  runTestpoint(mlp)
  -- showOutputDistributions(mlp)
end

setup()













