-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end

-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, target_grams, normalize, target_masks)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.targets = target_grams
  self.loss = 0
  
  self.target_masks = target_masks
  
  self.grams = {} 
  self.crits = {} 
  self.G = {}
  self.masked_inputs = {}

  for k = 1 , #self.target_masks do
    self.grams[k] = GramMatrix()
    self.crits[k] = nn.SmoothL1Criterion()
  end

end

function StyleLoss:updateOutput(input)
  -- Iterate through colors and update grams
  if not init then
    self.loss = 0
    for k , _ in ipairs(self.target_masks) do 
      self.masked_inputs[k] = torch.cmul(input,self.target_masks[k]:add_dummy():expandAs(input))

      self.G[k] = self.grams[k]:forward(self.masked_inputs[k])

      if(self.target_masks[k]:mean() > 0) then
        self.G[k]:div(input:nElement()*self.target_masks[k]:mean())
      end

      self.loss = self.loss + self.crits[k]:forward(self.G[k], self.targets[k])  
    end
  end 

  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  -- Iterate through colors and get gradient
  if not init then
    self.gradInput = gradOutput:clone():zero()

    for k , _ in ipairs(self.target_masks) do 
      local dG = self.crits[k]:backward(self.G[k], self.targets[k])
      
      if self.target_masks[k]:mean() > 0 then
        dG:div(input:nElement()*self.target_masks[k]:mean())
      end

      local gradInput = self.grams[k]:backward(self.masked_inputs[k], dG)
      if self.normalize then
        gradInput:div(torch.norm(gradInput, 1) + 1e-8)
      end
      self.gradInput:add(gradInput)

    end
    self.gradInput:add(gradOutput)
  end
  return self.gradInput
end