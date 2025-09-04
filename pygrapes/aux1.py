import torch
import numpy as np
import sys
def get_maximum_propdist(full_sim_size,voxel_size,lambda_val):
    Propagator_wave_dim = full_sim_size[0:2]
    propagator_wave_size = torch.tensor(Propagator_wave_dim) * torch.tensor(voxel_size)
    dist_picker_r = Propagator_wave_dim[0] * (voxel_size[0])**2 / lambda_val
    dist_picker_c = Propagator_wave_dim[1] * (voxel_size[1])**2 / lambda_val
    
    print('the maximum prop dist for these params =',"%5.5E" % (dist_picker_r),'m')
    
def get_maximum_propdist2(delta_res,lambda_val):
    z = (0.32 * (delta_res)**2) / lambda_val
    print('the maximum prop dist for these params =',"%5.5E" % (z),'m')
def rescale_0_1(input):
    imin = torch.min(input)
    imax = torch.max(input-imin)
    if imax == 0:
        output = input
    else:    
        output = (input-imin)/(imax)
    return output    
def print_progress_bar(iteration, total, bar_length=20):
    progress = (iteration / total)
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f'\rthis iter progress: [{arrow + spaces}] {int(progress * 100)}%')
    sys.stdout.flush()
def create_iso_random_initguess(xsize,ysize,zsize,aspr):
    str1 = torch.rand(xsize,int(ysize/aspr),zsize)
    str2 = torch.nn.functional.interpolate(str1.unsqueeze(0).unsqueeze(0),size=[xsize,ysize,zsize]).squeeze().unsqueeze(0)
    str3 = torch.tensor(gaussian_filter(str2.cpu().numpy(),5)).to(torch.float32)
    vec_X = torch.linspace(-2,2,str2.size(2))
    vec_Y = torch.linspace(-2,2,str2.size(1))
#testcurve_X = -2*(vec_X)**2+10
    testcurve_Y = 5*(vec_Y)**2
    test_structure_curve,_ = torch.meshgrid(testcurve_Y,vec_X)
#     plt.imshow(test_structure_curve.cpu(),aspect=0.01)
    
    
#     str3 += test_structure_curve
    return str3
    
def generate_per_spoke_modulated_siemens_star(
    width=512, height=512,
    num_spokes=36,
    r_inner=50, r_outer=200,
    modulated_spokes=None  # dict: {index: (r_inner_alt, r_outer_alt)}
):
    if modulated_spokes is None:
        modulated_spokes = {}

    # Create coordinate grid
    y, x = np.ogrid[:height, :width]
    cx, cy = width // 2, height // 2
    x = x - cx
    y = y - cy

    # Convert to polar
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta_norm = (theta + np.pi) / (2 * np.pi)

    # Spoke indices
    spoke_idx = (theta_norm * num_spokes).astype(int)

    # Alternating spoke pattern
    is_spoke = ((theta_norm * num_spokes) % 1) < 0.5

    # Initialize radius maps with default values
    r_outer_map = np.full_like(r, r_outer, dtype=float)
    r_inner_map = np.full_like(r, r_inner, dtype=float)

    # Apply custom values for each modulated spoke
    for idx, (ri_alt, ro_alt) in modulated_spokes.items():
        mask = spoke_idx == idx
        r_outer_map[mask] = ro_alt
        r_inner_map[mask] = ri_alt

    # Mask for radius within bounds
    radius_mask = (r >= r_inner_map) & (r <= r_outer_map)

    # Final Siemens star
    siemens_star = is_spoke & radius_mask

    return torch.tensor(siemens_star.astype(np.float32)).unsqueeze(0)


def create_GT_structure(width,height,bsx=1550,bsy=-170,downscaley = 1,downscalez = 1,height_voxels=2):
    image_tensor = torch.zeros(1, 1, height, width)  # 1 channel (grayscale), 1 batch
    for n1 in range(erclogo.shape[0]):
    
        patch_coords = (round((erclogo[n1,0])/(voxel_size[1]*downscaley).cpu().numpy()+bsx),
                        round((erclogo[n1,1])/(slab_thickness*downscalez).cpu().numpy()-((min(erclogo[:,1])/(slab_thickness*downscalez).cpu().numpy()+bsy))),
                        round((erclogo[n1,0]+erclogo[n1,2])/(voxel_size[1]*downscaley).cpu().numpy()+bsx),
                        round((erclogo[n1,1]+erclogo[n1,3])/(slab_thickness*downscalez).cpu().numpy()-((min(erclogo[:,1])/(slab_thickness*downscalez).cpu().numpy()+bsy))))
    
        mask = torch.zeros(1, 1, height, width)  # Initialize with zeros
        
        mask[:, :, patch_coords[1]:patch_coords[3], patch_coords[0]:patch_coords[2]] = 1
    
        # Apply the mask to the image tensor to create a binary image
        image_tensor = image_tensor + mask
        image_tensor = torch.clamp(image_tensor,0,1)
    image_tensor = image_tensor.squeeze().flip(0)*height_voxels
    return image_tensor.unsqueeze(0)




