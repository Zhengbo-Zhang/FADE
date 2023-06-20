from dcn_v2 import DCN
import torch

input = torch.randn(2, 64, 128, 128).cuda()
# wrap all things (offset and mask) in DCN
dcn = DCN(64, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2).cuda()
output = dcn(input)
print(output.shape)