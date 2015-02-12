require 'nn'

local EnergyContainer, parent = torch.class('nn.EnergyContainer', 'nn.Sequential')

function EnergyContainer:updateOutput(input)
  self.energy = 0
  local currentOutput = input
  for i=1,#self.modules do
   currentOutput = self.modules[i]:updateOutput(currentOutput)

   -- for i = 1,
   self.energy = self.energy + currentOutput:norm(0.5)
  end
  self.output = currentOutput
  return currentOutput
end

function EnergyContainer:updateGradInput(input, gradOutput)
  local currentGradOutput = gradOutput
  local currentModule = self.modules[#self.modules]
  for i=#self.modules-1,1,-1 do
    local previousModule = self.modules[i]
    currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
    currentModule = previousModule
  end
  currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
  self.gradInput = currentGradOutput
  return currentGradOutput
end