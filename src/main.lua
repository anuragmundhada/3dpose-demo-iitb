require 'hdf5'
require 'csvigo'
require 'cunn'
require 'cudnn'
require 'util.lua'
require 'optim'
require 'nngraph'
require 'image'

cmd = torch.CmdLine()

cmd:option('-gpu',1,'Use gpu')
cmd:option('-imagePath', '../data/', ' Path of predictions of ff model ')
cmd:option('-savePreds', '', 'path to csv file if you want to save the predictions ')
cmd:option('-convModel', '../models/unit-pose-net.t7', 'Path of the single image model')
cmd:option('-temporalModel', '../models/time-pose-net.t7', ' Path of the temporal model')
cmd:option('-skelFit', 1, 'Fit a standard skeleton to predicted poses')
cmd:option('-display', 1, 'Display the poses')
cmd:option('-exp','default','Name of experiment')

opt = cmd:parse(arg)

paths.dofile('model.lua')
paths.dofile('util.lua')
paths.dofile('img.lua')
paths.dofile('pyTools.lua')

-- Read in files from the image directory
f = io.popen('ls ' .. opt.imagePath)

-- buffer of past frames for the temporal model (stores)
local framebuffer = {}
local outputs = {}
count = 1
for img in f:lines() do
    print(img)
	-- Processing image for convnet
    local img = image.load(opt.imagePath..img):narrow(1, 1, 3)
    local h, w = img:size(2), img:size(3)
    local c = torch.Tensor({w / 2, h / 2})
    local size = math.max(h, w)
    local inp = crop(img, c, 1 * size / 200.0, 0, 256)
 
    -- Individual pose estimates per frame
    local output = convModel:forward(inp:view(1, 3, 256, 256):cuda())
    local pred = convert_heatmap_to_worldcoor(output[3], output[2]):cuda() -- converting to world coordinates from heatmap represenation
    table.insert(framebuffer, pred:reshape(48))

    -- Temporal model kicks in when we have pose estimates for 20 frames
    if count >= 20 then
      pred = timeModel:forward(torch.cat(framebuffer))
      table.remove(framebuffer,1)
    end

    -- Fit a standard skeleton by preserving bone directions
    pred = pred:reshape(16,3)
    if opt.skelFit then
    	pred = fit_standard_skeleton(pred)
    end

    if opt.display and count % 20 ==0 then
    	pyFunc('Show3d', {joint=4*pred:float(), img=img, noPause = torch.zeros(1), id = torch.ones(1) * 1})
    end
    if opt.savePreds ~= '' then
    	pred = pred:reshape(48)
    	table.insert(outputs,torch.totable(pred))
    end

    count = count + 1
end

if opt.savePreds ~= '' then
    print("Saving predictions")
	csvigo.save({path = '../results/'..opt.exp..'.csv', data = outputs, verbose = true})
end