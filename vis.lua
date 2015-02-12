function lines(str)
  local t = {}
  local function helper(line) table.insert(t, line) return "" end
  helper((str:gsub("(.-)\r?\n", helper)))
  return t
end

function showOutputDistribution(network)
  local numDecimals = 2
  local digitFactor = 10 ^ numDecimals
  local distSize = 20
  -- local dist = torch.histc(torch.abs(network.modules[1].output), distSize)
  -- local numbers = lines(tostring(dist:reshape(1, distSize)))
  local numbers = lines(tostring(torch.abs(network.modules[1].output)))

  local output = ""
  for i, line in ipairs(numbers) do
    if i < #numbers -1 then
      output = output.."\n"..line
    end
  end

  os.execute('echo "'..output..'" | histogram.py -b 10 --no-mvsd --min=0 --max='.. 1.1 * torch.max(torch.abs(network.modules[1].output)))
  -- os.execute('echo '..numbers[1]..' | distribution --tokenize=word')
  -- os.execute('spark '..numbers[1])
  -- print(0,
  --       "",
  --       torch.round(torch.max(torch.abs(network.modules[1].output)) * digitFactor) / digitFactor)
end

function showOutputDistributions(...)
  local networks = table.pack(...)
  for i, network in ipairs(networks) do
    print("\noutput distribution #"..tostring(i)..":")
    showOutputDistribution(network)
  end
end
