import torch
import numpy as np
def fill_3d_tensor(A,structure_height,params_size,oversample_structure,oversample_factor):

    # Get the dimensions of the 2D tensor
    width,height = A.shape
    output3d = A.repeat(structure_height,1,1)-torch.arange(structure_height).unsqueeze(1).unsqueeze(1).expand(structure_height,width,height)
    output3d = torch.clamp(output3d,0,1)
    if oversample_structure == True and oversample_factor !=1:
        weight = torch.ones(1,1,1,1,oversample_factor)/oversample_factor
    
        output3d = torch.nn.functional.conv3d(output3d.unsqueeze(0).unsqueeze(0), weight, stride=(1,1,oversample_factor), padding=0, dilation=1, groups=1).squeeze()

    if not output3d.size(1) == params_size[1]:
        sd = output3d.size(1) - params_size[1]
        if sd > 0:
            output3d = output3d[:,:-int(sd),:]
        if sd < 0:
            output3d = torch.nn.functional.interpolate(output3d.unsqueeze(0).unsqueeze(0), size=params_size, mode='trilinear').squeeze()
    
    return output3d.flip(0)

def interp_3d_tensor(A,structure_height,oversample_structure,oversample_factor):

    if oversample_structure == True:
        # weight = torch.ones(1,1,1,1,oversample_factor)/oversample_factor
    
        # output3d = torch.nn.functional.conv3d(A.unsqueeze(0).unsqueeze(0), weight, stride=(1,1,oversample_factor), padding=0, dilation=1, groups=1).squeeze()
        output3d = A # pooler3d = torch.nn.AvgPool1d(kernel_size=(oversample_factor))
        # output3d = pooler3d(output3d)
    if not output3d.size(1) == params_size[1]:
        sd = output3d.size(1) - params_size[1]
        if sd > 0:
            output3d = output3d[:,:-int(sd),:]
        if sd < 0:
            output3d = torch.nn.functional.interpolate(output3d.unsqueeze(0).unsqueeze(0), size=params_size, mode='trilinear').squeeze()
    
    return output3d

def display_3d_tensor(A,structure_height,params_size,oversample_structure,oversample_factor):
	width,height = A.shape
        output3d = A.repeat(structure_height,1,1)-torch.arange(structure_height).unsqueeze(1).unsqueeze(1).expand(structure_height,width,height)
        output3d = torch.clamp(output3d,0,1)
        



