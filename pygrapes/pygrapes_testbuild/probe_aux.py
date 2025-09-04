import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from .propagation import optimise_probe_prop , create_freq_mesh,wave_interact_partial_voxels

def probe_tilt_gradient(probe,inc_angle,voxel_size,slab_thickness,wavelength):
        
        phase_gradient_Y = torch.linspace(0,voxel_size[0]*probe.shape[0]*torch.tan(torch.deg2rad(torch.tensor(inc_angle))),probe.shape[0])
        phase_gradient_X = torch.linspace(0,voxel_size[1]*probe.shape[1]*torch.tan(torch.deg2rad(torch.tensor(inc_angle))),probe.shape[1])
        
        PhaserampmatY, _ = torch.meshgrid(-phase_gradient_Y,phase_gradient_X)

        probe_amp = torch.abs(probe)
        probe_phase = torch.angle(probe)
        
        tilted_probe = probe_amp * torch.exp(1j*(probe_phase-(2*torch.pi*(PhaserampmatY/wavelength))))
        
        return tilted_probe
  
def flux_rescale(probe_in, flux_in): 
    current_flux = torch.sum(torch.abs(probe_in) ** 2)
    flux_rescale_factor = flux_in / current_flux
    probe_out = (flux_rescale_factor)**2 * torch.abs(probe_in) * (torch.cos(torch.angle(probe_in)) + 1j * torch.sin(torch.angle(probe_in)))
    return probe_out
    
def retain_flux(previous_amp,previous_phase, current_amp, current_phase):
    current_probe = current_amp * torch.exp(1j*current_phase)
    previous_probe = previous_amp * torch.exp(1j*previous_phase)
    current_flux = torch.sum(torch.abs(current_probe) ** 2)
    previous_flux = torch.sum(torch.abs(previous_probe) ** 2)
    flux_rescale_factor = current_flux / previous_flux
    current_probe_rescaled  = (flux_rescale_factor)**2 * torch.abs(current_probe) * (torch.cos(torch.angle(current_probe)) + 1j * torch.sin(torch.angle(current_probe)))
    amp = torch.abs(current_probe_rescaled)
    phase = torch.angle(current_probe_rescaled)
    
    return amp, phase

def make_prop_mask(inputwave,rad,blur,aspr):
    if type(aspr) == torch.Tensor:
        aspr = aspr.cpu().numpy()
        
    inputsize = inputwave.size()
    propmaskinit = np.zeros(inputsize)
    mask_size = inputsize

    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(mask_size[1])-mask_size[1]/2, (np.arange(mask_size[0])-mask_size[0]/2))

    # Calculate the distance from the center of the circle
    distance = np.sqrt((x)**2 + (y/aspr)**2)

    # Create a binary circle mask
    circle_mask = np.where(distance <= rad, 1, 0)
    circle_mask = circle_mask.astype(float)
    # propmaskinit[30:-30,30:-30] = 1
    sigma = blur
    propmask_b = torch.tensor(gaussian_filter(circle_mask,sigma)).to(torch.float32)

    # propmask_b = torch.tensor(circle_mask).to(torch.float32)

    propmask_b[propmask_b>1] = 1
    return propmask_b

def get_tf_modelprobe(propdist,lambda_val,voxel_size,grid_shape):
    u,v = create_freq_mesh(voxel_size,grid_shape)
    u = u/voxel_size[0]
    v = v/voxel_size[1]
    H = torch.exp(-1j*torch.pi*lambda_val*propdist*(u**2+v**2))
    return H

    
def make_csaxs_model_probe(wavelength,probe_dims,desired_pixel_size,
                           probe_diameter,central_stop_diameter,zone_plate_diameter,
                          outer_zone_width,prop_dist):
    
    zp_f = zone_plate_diameter*outer_zone_width/wavelength
#     print("zp_f:", zp_f)
    upsample = 10
    voxel_ratio = desired_pixel_size[1]/desired_pixel_size[0]
    
    
    defocus = prop_dist
    Nprobe = upsample*torch.tensor(probe_dims)
    padsize = int(((Nprobe[0]*(voxel_ratio-1))/2))
#     print("guessed pad:", padsize)
#     print("desired pixel size before psi rando trans",desired_pixel_size)
    desired_pixel_size = (zp_f + defocus) * wavelength / (Nprobe*desired_pixel_size)
#     print("desired pixel size after psi rando trans",desired_pixel_size)
    r1_pix = probe_diameter/desired_pixel_size
#     print(r1_pix)
    r2_pix = central_stop_diameter/desired_pixel_size
    xvec = torch.arange(-Nprobe[1]/2,torch.floor((Nprobe[1]-1)/2))
    yvec = torch.arange(-Nprobe[1]/2,torch.floor((Nprobe[1]-1)/2))
    x,y = torch.meshgrid(xvec,yvec)
    r2 = (x*desired_pixel_size[1])**2+(y*desired_pixel_size[1])**2
    w = make_prop_mask(r2,(r1_pix[1]/2).cpu().numpy(),0.1,1)
    w += -make_prop_mask(r2,(r2_pix[1]/2).cpu().numpy(),0.1,1)
    tf = torch.exp(-1j*torch.pi*(r2)/(wavelength*zp_f))
    wc = w*tf
#     probe_hr1 = farfield_PSI_prop(wc,wavelength,zp_f+defocus,desired_pixel_size)
    
    N = Nprobe
    
    wcp = torch.nn.functional.pad(wc,(0,0,padsize,padsize),mode='constant',value=0)
    r2p = torch.nn.functional.pad(r2,(0,0,padsize,padsize),mode='constant',value=0)
    propdist = (zp_f+defocus)
    probe_hr1 = -1j * (torch.exp(1j * torch.pi * wavelength * propdist * r2p / (N[0]*N[1])) 
                       * torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(
                           wcp * torch.exp(1j * torch.pi * r2p / (wavelength*propdist))))))
    
    
    phrs = torch.tensor((probe_hr1.size(0)/2,probe_hr1.size(1)/2))
    cropinds = (int(float(phrs[0])-float((probe_dims[0]+padsize*2/upsample)/2)),
                int(float(phrs[0])+float((probe_dims[0]+padsize*2/upsample)/2)),
                int(float(phrs[1])-float((probe_dims[1])/2)),
                int(float(phrs[1])+float((probe_dims[1])/2)),
               )
#     print(cropinds)
    model_probe1 = probe_hr1[cropinds[0]:cropinds[1],cropinds[2]:cropinds[3]]
    return model_probe1

def probe_subpixel_shift(inputprobe,shift_amount_vert,shift_amount_hor,slab_thickness,y_voxel_size):
    
    # the amount of y shift corresponds to 1 pixel 
    # up or down is 1 slice further or closer along the z direction.
    #so if we wish to shift by 500nm and hte slab thickness is 1000nm, the corresponding shift is 0.5 pixels.
    shift_amount_pixelsv = shift_amount_vert 
    shift_amount_normalisedv = shift_amount_pixelsv / (inputprobe.size(0)*0.5)
    shift_amount_pixelsh = shift_amount_hor 
    shift_amount_normalisedh = shift_amount_pixelsh / (inputprobe.size(1)*0.5)
    
    input_tensor = torch.zeros(1,2,inputprobe.size(0),inputprobe.size(1))
    input_tensor[:,0,:,:] = torch.real(inputprobe) 
    input_tensor[:,1,:,:] = torch.imag(inputprobe)
    
    inputsize = input_tensor.size()
#     print(inputsize)
    grid = torch.zeros(inputsize[0],inputsize[2],inputsize[3],2)
    new_y  = torch.linspace(-1,1,inputprobe.size(0)) + shift_amount_normalisedv
    new_x  = torch.linspace(-1,1,inputprobe.size(1)) + shift_amount_normalisedh
    newx_mesh,newy_mesh = torch.meshgrid(new_y,new_x)
    grid[:,:,:,0] = newy_mesh
    grid[:,:,:,1] = newx_mesh
    output = torch.nn.functional.grid_sample(input_tensor,grid,align_corners=True,mode="bilinear").squeeze()
    output_complex = (output[0,:,:] + 1j*output[1,:,:]).to(torch.complex64)
#     print(output_complex.size())
    return output_complex
def probe_subpixel_shift_fourier(inputprobe,shift_amount_vert,shift_amount_hor,slab_thickness,voxel_size,hx_shift=0,hz_shift=0):
    probe_F = torch.fft.fft2(inputprobe)
    xx,yy = create_freq_mesh(voxel_size,probe_F.size())
    m1 = torch.exp(-1j*2*torch.pi*(xx*(shift_amount_hor+hx_shift)+yy*(shift_amount_vert+hz_shift)))
    # print(torch.sum(torch.isnan(m1)))
    probe_shifted = torch.fft.ifft2(probe_F*m1)
    return probe_shifted
 
def generate_model_probe(
    lambda_val,
    voxel_size,
    crop_window_size,
    scan_identifier_list,
    new_flux=4.0e-8,
    probe_vmax=8e-4,
    recon_FFT_vmin=-5,
    recon_FFT_vmax=0,
    new_prop=-8.5e-4,
    probepad=400,
    initprobesize=[182, 182],
    probecenter=None,
    probesigma_x=30.0,
    probesigma_y=100.0,
    probe_diameter=240e-6,
    central_stop_diameter=48e-6,
    zone_plate_diameter=490e-6,
    outer_zone_width=43e-9
):


    probe_aspr = voxel_size[0] / voxel_size[1]
    cd1 = int((initprobesize[0] - crop_window_size[1]) / 2)
    n_probe_modes = int(np.max(scan_identifier_list) + 1)
    model_probe_pixel_size = voxel_size

    init_model_probe = make_csaxs_model_probe(
        lambda_val,
        initprobesize,
        model_probe_pixel_size,
        probe_diameter,
        central_stop_diameter,
        zone_plate_diameter,
        outer_zone_width,
        new_prop
    )

    init_model_probe = optimise_probe_prop(
        torch.nn.functional.pad(init_model_probe, (probepad, probepad, probepad, probepad)),
        new_prop,
        lambda_val,
        model_probe_pixel_size
    )[probepad:-probepad, probepad:-probepad] * new_flux

    model_probes = torch.zeros(
        init_model_probe.size(0),
        init_model_probe.size(1),
        n_probe_modes,
        dtype=torch.complex64
    )

    model_probes[:, :, 0] = init_model_probe  # first mode is voxel-size matched

    for n1 in range(n_probe_modes):
        print("probe mode:", n1)
        this_model_probe = make_csaxs_model_probe(
            lambda_val,
            initprobesize,
            model_probe_pixel_size,
            probe_diameter,
            central_stop_diameter,
            zone_plate_diameter,
            outer_zone_width,
            new_prop
        )

        newsd = int((init_model_probe.size(0) - this_model_probe.size(0)) / 2)
        this_model_probe = torch.nn.functional.pad(this_model_probe, (0, 0, newsd, newsd), value=0)

        if this_model_probe.size() != model_probes[:, :, n1].size():
            this_model_probe = torch.nn.functional.pad(this_model_probe, (0, 0, 0, 1), value=0)

        this_model_probe = optimise_probe_prop(
            torch.nn.functional.pad(this_model_probe, (probepad, probepad, probepad, probepad)),
            new_prop,
            lambda_val,
            model_probe_pixel_size
        )[probepad:-probepad, probepad:-probepad]

        this_model_probe *= new_flux
        model_probes[:, :, n1] = this_model_probe

    orig_csaxs_probe = init_model_probe * new_flux
    print("using model probe, with dimensions of", init_model_probe.size())

    return orig_csaxs_probe, model_probes


#usage 
# orig_csaxs_probe, model_probes = generate_model_probe(
#     lambda_val=lambda_val,
#     voxel_size=voxel_size,
#     crop_window_size=crop_window_size,
#     scan_identifier_list=scan_identifier_list
# )


def combine_probe_modes(multimode_probe_in,scan_number,scan_identifier_list):
        
    if len(multimode_probe_in.size()) == 3:
        combined_probes = multimode_probe_in[:,:,int(scan_identifier_list[scan_number])]
    if len(multimode_probe_in.size()) == 2:
        combined_probes = multimode_probe_in
    return combined_probes


