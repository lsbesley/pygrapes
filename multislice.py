import torch
import numpy as np
from .propagation import get_tf_longprop ,wave_interact_partial_voxels,longprop
from .probe_aux import probe_tilt_gradient,probe_subpixel_shift_fourier
def MSForward_GI_SS_novol_partialvoxel_pre(xr,wavelength,full_sim_size,params_size,voxel_size,slab_thickness,inc_angle,slab_pad_pre,slab_pad_post,probe,probe_buffer,probe_substrate_buffer,wavemask,shift_amt,substrate_layers,init_xr_substrateamt,top_buffer,subpixel_shift_y,subpixel_shift_z,hx_shift,hz_shift,ndelta,nbeta):
        
        old_max = torch.max(torch.abs(torch.fft.fft2(probe)))
        substrate_layers_pre = substrate_layers
        substrate_start = (full_sim_size[0]-substrate_layers_pre)*voxel_size[0]
#         probe_substrate_buffer = 20
        probe_in = probe
        probe_halfsize = (probe_in.size(0)*voxel_size[0])/2
        probe_insertion_Y = int((full_sim_size[1] - probe_in.size(1))/2)
        reflection_distance = 2*(probe_halfsize+probe_substrate_buffer*voxel_size[0])/(torch.tan(torch.deg2rad(torch.tensor(inc_angle))))
        pre_post_amt = (reflection_distance - (xr.size(2)*slab_thickness)) / 2
        new_c_wave_size = probe_in.size(0)+substrate_layers_pre+probe_substrate_buffer
        if new_c_wave_size < full_sim_size[0]:
            new_c_wave_size = full_sim_size[0]
        c_wave = torch.complex(torch.zeros((new_c_wave_size),full_sim_size[1]),torch.zeros((new_c_wave_size),full_sim_size[1])).to(torch.complex64)

        if pre_post_amt<0:
            pre_post_amt = 1e-6
        
        n_slices = 50
        pre_prop_dist = pre_post_amt/n_slices
        newphantom = torch.zeros(c_wave.size(0),c_wave.size(1))
        newphantom[-int(substrate_layers_pre):,:] = 1

        tf2 = get_tf_longprop(pre_prop_dist,wavelength,voxel_size,c_wave.size())
        probe_insertion_x_coords = ((substrate_layers+probe_substrate_buffer+probe_in.size(0)),(substrate_layers+probe_substrate_buffer))
        if probe_insertion_x_coords[1] < substrate_layers:
            probe_cutoff_amount = substrate_layers - probe_insertion_x_coords[1]
            probe_in[-probe_cutoff_amount:,:] = 0
        c_wave[-probe_insertion_x_coords[0]:-probe_insertion_x_coords[1],probe_insertion_Y:int(probe_insertion_Y+probe.size(1))] = probe_tilt_gradient(probe_in,inc_angle,voxel_size,pre_prop_dist,wavelength)
        c_wave = probe_subpixel_shift_fourier(c_wave,subpixel_shift_z,subpixel_shift_y,slab_thickness,voxel_size,hx_shift,hz_shift)
        
        for n1 in range(n_slices):
            c_wave = wave_interact_partial_voxels(c_wave,newphantom,pre_prop_dist,wavelength,ndelta,nbeta)
            c_wave = longprop(c_wave,tf2)#wave_propagate_2d_faster_usefilter(c_wave,tf2,1)#
        
        c_wave = torch.fft.ifft2(torch.fft.fft2(c_wave))
        c_wave = c_wave[-int(full_sim_size[0]):,:]
        

        
        
    
        return c_wave,pre_post_amt,probe_insertion_Y
        
        
def MSForward_GI_SS_novol_partialvoxel_mid(xr,wavelength,cwavein,full_sim_size,params_size,voxel_size,slab_thickness,inc_angle,substrate_layers,ndelta,nbeta):
        c_wave = cwavein
        
#         tf = get_tf_nearfield(slab_thickness,lambda_val,voxel_size,c_wave.size())
        tf = get_tf_longprop(slab_thickness,wavelength,voxel_size,c_wave.size())
        upscale_slices = int(torch.round(torch.tensor(full_sim_size[2]/xr.size(2))))
        upscale_slices_x = int(torch.round(torch.tensor(params_size[0]/xr.size(0))))
        us_size = (params_size[0],params_size[1])
        midslice = torch.zeros(c_wave.size(1),xr.size(2))
        sideslice = torch.zeros(c_wave.size(0),xr.size(2))
        for n1 in range(xr.size(2)):
            S1 = xr[:,:,n1]

            this_slab = torch.nn.functional.pad(torch.nn.functional.pad(S1,(0,0,int(full_sim_size[0]-S1.size(0)-substrate_layers),0),value=0),(0,0,0,substrate_layers),value=1)

            for n2 in range(upscale_slices):
                if (torch.cuda.memory_allocated() / (1024**2)) > 24000:

                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        c_wave = wave_interact_partial_voxels(c_wave,this_slab,slab_thickness,wavelength,ndelta,nbeta)
                        c_wave = longprop(c_wave,tf)#wave_propagate_2d_faster_usefilter(c_wave,tf,1)
#                         c_wave=c_wave*wavemask
                else:
                    c_wave = wave_interact_partial_voxels(c_wave, this_slab, slab_thickness, wavelength,ndelta,nbeta)

                    c_wave = longprop(c_wave,tf)

                with torch.no_grad():
                    midslice[:,n1] = torch.abs(c_wave[-substrate_layers,:])
                    sideslice[:,n1] = torch.abs(c_wave[:,int(full_sim_size[1]/2)])

        return c_wave, midslice,sideslice
        
def MSForward_GI_SS_novol_partialvoxel_post(xr,wavelength,cwavein,full_sim_size,params_size,voxel_size,slab_thickness,inc_angle,substrate_layers,prepostdist,post_prop_dist_mult,probe_insertion_Y,probesize,init_xr_substrateamt,ndelta,nbeta):
        n_slices = 100
        side_buffer = 50
        pre_prop_dist = (prepostdist*post_prop_dist_mult)/n_slices
        top_x_post = 200    
        top_buffer = top_x_post
        substrate_layers_pre = substrate_layers
        c_wave = torch.nn.functional.pad(cwavein,(side_buffer,side_buffer,top_buffer,0),value=0)
        
        
        newphantom = torch.zeros(c_wave.size(0),c_wave.size(1))
        newphantom[-int(substrate_layers_pre):,:] = 1
        tf2 = get_tf_longprop(pre_prop_dist,wavelength,voxel_size,c_wave.size())
        for n1 in range(n_slices):
            c_wave = wave_interact_partial_voxels(c_wave,newphantom,pre_prop_dist,wavelength,ndelta,nbeta)
            c_wave = longprop(c_wave,tf2)
        c_wave = probe_tilt_gradient(c_wave,inc_angle,voxel_size,pre_prop_dist,wavelength)

        return c_wave
    
def multislice_3stage(this_xr,
                full_sim_size, params_size, voxel_size, slab_thickness,wavelength,
                inc_angle, slab_pad_pre, slab_pad_post,post_prop_dist_mult, probes_in_shifted,
                probe_buffer,probe_substrate_buffer,wave_deletion_mask,shift_amount,substrate_layers,init_xr_substrateamt,top_buffer,blankew,subpixel_shift_y,subpixel_shift_z,hx_shift,hz_shift,ndelta,nbeta):
    probesize = probes_in_shifted.size()
    
                        
    pre_EW,pre_post_amt,probe_insertion_Y = MSForward_GI_SS_novol_partialvoxel_pre(
                        this_xr,wavelength,
                        full_sim_size, params_size, voxel_size, slab_thickness,
                        inc_angle, slab_pad_pre, slab_pad_post, probes_in_shifted,
                        probe_buffer,probe_substrate_buffer,1,0,substrate_layers,init_xr_substrateamt,0,subpixel_shift_y,subpixel_shift_z,hx_shift,hz_shift,ndelta,nbeta)
    midEW,midslice,sideslice = MSForward_GI_SS_novol_partialvoxel_mid(this_xr,wavelength,pre_EW,
                   full_sim_size,params_size,voxel_size,slab_thickness,inc_angle,(substrate_layers),ndelta,nbeta)

    exit_wave = MSForward_GI_SS_novol_partialvoxel_post(this_xr,wavelength,midEW,
                    full_sim_size,params_size,voxel_size,slab_thickness,inc_angle,(substrate_layers),pre_post_amt,post_prop_dist_mult,probe_insertion_Y,probesize,init_xr_substrateamt,ndelta,nbeta)

    F1 = torch.abs(torch.fft.fftshift(torch.fft.fft2(exit_wave.flip(0))))
    F1 = torch.nn.functional.interpolate(F1.unsqueeze(0).unsqueeze(0),size=(probesize),mode='bilinear').squeeze()

    return F1,midslice




