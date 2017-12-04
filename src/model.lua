if opt.loadModel ~= 'none' then
    print('==> Loading convnet model from: ' .. opt.convModel.. ' and temporal model from '..opt.temporalModel)
    convModel = torch.load(opt.convModel)
    timeModel = torch.load(opt.temporalModel)
else
  assert(false, "Give the model path")
end

if opt.gpu == 1 then
  convModel:cuda()
  timeModel:cuda()
end

convModel:evaluate()
timeModel:evaluate()