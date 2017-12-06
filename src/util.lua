tree = { 
     {7, 0, 0}, {8,7,239.70}, {9,8,254.46}, {10,9,178.07}, 
     {13,9,150.85}, {12,13,279.66}, {11,12,246.43}, {14,9,150.85}, {15,14,279.66}, {16,15,246.43},
     {3,7,139.59}, {2, 3,448.04}, {1, 2,436.62}, {4, 7,139.59}, {5, 4,448.04}, {6, 5,436.62}
   }   

skeletonRef = {{1,2,1},    {2,3,1},    {3,7,1},
                    {4,5,2},    {4,7,2},    {5,6,2},
                    {7,9,0},    {9,10,0},
                    {13,9,3},   {11,12,3},  {12,13,3},
                    {14,9,4},   {14,15,4},  {15,16,4}}

function fit_standard_skeleton(pred)
    output = pred:clone():fill(0)
    for j = 1, #tree do
      local parent = tree[j][2]
      local joint = tree[j][1]
      local bone_length = tree[j][3]
      if parent == 0 then -- root
        output[joint] = pred[joint]
      else
      -- print(joint,parent, torch.norm(pred[joint] - target[parent], 2, 1):squeeze()/ torch.norm(target[joint] - target[parent], 2, 1):squeeze())
        local vecnorm = (pred[joint] - pred[parent]) / (torch.norm(pred[joint] - pred[parent], 2, 1):squeeze())
        output[joint] = output[parent] + bone_length*vecnorm
      end
    end
    return output
end

function convert_heatmap_to_worldcoor(outputReg, outputHM)
    local Reg = outputReg
    local tmpOutput = outputHM
    Reg = Reg:view(Reg:size(1), 16, 1)
    local z = (Reg + 1) * 64 / 2 
    -- p is a [torch.FloatTensor of size batchSizex16x2]
    local p = getPreds(tmpOutput):cuda()
    for i = 1,p:size(1) do
        for j = 1,p:size(2) do
            local hm = tmpOutput[i][j]
            local pX,pY = p[i][j][1], p[i][j][2]
            if pX > 1 and pX < 64 and pY > 1 and pY < 64 then
               local diff = torch.CudaTensor({hm[pY][pX+1]-hm[pY][pX-1], hm[pY+1][pX]-hm[pY-1][pX]})
               p[i][j]:add(diff:sign():mul(.25))
            end
        end
    end
    p:add(0.5)

    local pred = torch.zeros(16, 3)
    for j = 1, 16 do
        pred[j][1], pred[j][2], pred[j][3] = p[1][j][1], p[1][j][2], z[1][j][1]
    end
        
    local len_pred = 0
    local len_gt = 0
    for j = 1, #skeletonRef do
        len_pred = len_pred +  ((pred[skeletonRef[j][1]][1] - pred[skeletonRef[j][2]][1]) ^ 2 + 
                                (pred[skeletonRef[j][1]][2] - pred[skeletonRef[j][2]][2]) ^ 2 + 
                                (pred[skeletonRef[j][1]][3] - pred[skeletonRef[j][2]][3]) ^ 2) ^ 0.5
    end

    len_gt = 4296.99233013
    local root = 7
    local proot = pred[root]:clone()
    for j = 1, 16 do
        pred[j][1] = (pred[j][1] - proot[1]) / len_pred * len_gt
        pred[j][2] = (pred[j][2] - proot[2]) / len_pred * len_gt
        pred[j][3] = (pred[j][3] - proot[3]) / len_pred * len_gt
    end
    
    -- Fix torsor annotation descrepancy between h36m and mpii
    pred[8] = (pred[7] + pred[9]) / 2

    return pred
end

function getPreds(hm)
    assert(hm:size():size() == 4, 'Input must be 4-D tensor')
    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)
    preds[{{}, {}, 2}]:add(-1):div(hm:size(3)):floor():add(1)
    
    local predMask = max:gt(0):repeatTensor(1, 1, 2):float()
    preds:add(-1):cmul(predMask):add(1)
    
    return preds
end

function saveData(dict, tmpFile)
    local file = hdf5.open(tmpFile, 'w')
    for k, v in pairs(dict) do 
        file:write(k, v)
    end
    file:close()
end