import torch
import numpy as np
from .probe_aux import make_prop_mask
def zero_storage_tensors():
    probes_cumulative_grad = torch.zeros_like(probes_param)
    spsz_cumulative = torch.zeros_like(subpixel_shifts_z)
    spsy_cumulative = torch.zeros_like(subpixel_shifts_y)
    hx_cumulative = torch.zeros_like(hx_shift)
    hz_cumulative = torch.zeros_like(hz_shift)
def zero_grads(optimizer,optimizer_adam,probe_optimizer,prop_optim,optimize_scan_offsets,scan_positions_optim,use_noise,noise_optim):    
        optimizer.zero_grad()
        optimizer_adam.zero_grad()
        probe_optimizer.zero_grad()
        prop_optim.zero_grad()
        if optimize_scan_offsets ==  1 :
            scan_positions_optim.zero_grad()
        if use_noise == 1:
            noise_optim.zero_grad()

def apply_tv(xr,tvx_alpha,tvy_alpha,tvz_alpha,tvt_alpha,i,iii,tv_start):
    if i < tv_start:
            tvx = 0
            tvy = 0
            tvz = 0
            tvsy = 0
            tvsz = 0
#             voxel_weights = 0
    else:

        if use_per_scan_TV == True:
            tvx = 0 #tvx is zero in height map optimisation.
        else:
            tvx = 0 #tvx is zero in height map optimisation.
        if use_per_scan_TV == True:
            tvy = torch.nansum(torch.abs(torch.diff(xr[0,int(new_yposindex[iii][0]+yzpad):int(new_yposindex[iii][1]+yzpad),
               int(new_zposindex[iii][0]+yzpad):int(new_zposindex[iii][1]+yzpad)],dim=1)**2))*tvy_alpha 
            tvz = torch.nansum(torch.abs(torch.diff(xr[0,int(new_yposindex[iii][0]+yzpad):int(new_yposindex[iii][1]+yzpad),
               int(new_zposindex[iii][0]+yzpad):int(new_zposindex[iii][1]+yzpad)],dim=2)**2))*tvz_alpha 
        else: 
            tvy = torch.nanmean(torch.abs(torch.diff(xr[0,:,:],n=1,dim=0)**2))**0.5*tvy_alpha 
            tvz = torch.nanmean(torch.abs(torch.diff(xr[0,:,:],n=1,dim=1)**2))**0.5*tvz_alpha 
    tv_total = (tvx+tvy+tvz)*tvt_alpha
    
    return tvx,tvy,tvz,tv_total

def store_xr_gradients(xr,xrg, iii, new_yposindex, new_zposindex, yzpad, i,do_gradient_accumulation):
    with torch.no_grad():
        y0, y1 = int(new_yposindex[iii][0] + yzpad), int(new_yposindex[iii][1] + yzpad)
        z0, z1 = int(new_zposindex[iii][0] + yzpad), int(new_zposindex[iii][1] + yzpad)
        
        if do_gradient_accumulation:
            this_xrg = xr.grad[:, y0:y1, z0:z1]
            this_xrg[torch.isnan(this_xrg)] = 0
            xrg[:, y0:y1, z0:z1] += this_xrg
        else:
            this_xrg = xr.grad[0, y0:y1, z0:z1]
            # optimizer.param_groups[0]['lr'] = grads_target / (torch.mean(this_xrg) + torch.std(this_xrg))

    if not do_gradient_accumulation:
        if i < 500:
            optimizer.step()
        else:
            optimizer_adam.step()
        xr.data = torch.clamp(xr.data, 0, params_size[0])

def store_scan_offset_gradients(subpixel_shifts_z,subpixel_shifts_y,hx_shift,hz_shift,optimize_scan_offsets,i):
    if optimize_scan_offsets ==  1 :
            if i > 10:
                spsz_cumulative += subpixel_shifts_z.grad
                spsy_cumulative += subpixel_shifts_y.grad
                hx_cumulative += hx_shift.grad
                hz_cumulative += hz_shift.grad
        

def store_probe_gradients():
    with torch.no_grad():
            probes_cumulative_grad += probes_param.grad
            probes_cumulative_grad[torch.isnan(probes_cumulative_grad)] = 0


def store_noise_gradients(noise_optim,noise_guess,use_noise,noise_mean,noise_std):
    if use_noise == 1:
        noise_optim.step()
        #noise is modelled as a normal distribution, and so we clip it to within 3 std of mean.
        noise_guess.data = torch.clamp(noise_guess.data,0,(noise_mean+3*noise_std))

def apply_gradients_and_update(optimizer,optimizer_adam,xr,xrg,beamfootprints,params_size,beam_footprints_binary,do_gradient_accumulation,using_n_scans_per_pixel,i, spsy_cumulative,spsz_cumulative,hx_cumulative,hz_cumulative,subpixel_shifts_y,subpixel_shifts_z,hx_shift,hz_shift,probes_param,probes_cumulative_grad,num_scans):
    if do_gradient_accumulation == True:
        optimizer.zero_grad()
        optimizer_adam.zero_grad()
        if using_n_scans_per_pixel == True:
            xr.grad = xrg/beamfootprints #warning: cant also be in the previous bit of code... 
        else:        
            xr.grad = xrg

        if i< 500:
            optimizer.step()
        else:
            optimizer_adam.step()
        
        with torch.no_grad():
            xr.data = torch.clamp(xr.data,0,params_size[0])*beam_footprints_binary
            xr.data[torch.isnan(xr.data)] = 0
        
            
    subpixel_shifts_y.grad = spsy_cumulative
    subpixel_shifts_z.grad = spsz_cumulative
    hx_shift.grad = hx_cumulative 
    hz_shift.grad = hz_cumulative 
    # scan_positions_optim.step()
    subpixel_shifts_z.data = torch.clamp(subpixel_shifts_z.data,-2e-7,2e-7)
    subpixel_shifts_y.data = torch.clamp(subpixel_shifts_y.data,-2e-6,2e-6)
    hx_shift.data = torch.clamp(hx_shift.data,-2e-6,2e-6)
    hz_shift.data = torch.clamp(hz_shift.data,-2e-7,2e-7)
    #transfer probe cumulative gra ot the actual tensor.
    probes_param.grad = probes_cumulative_grad/num_scans

def apply_probe_constraints(probestart,probe_optimizer,probe_grads_target,probes_cumulative_grad,num_scans,i,probes_param,psr):
    if i > probestart:
        with torch.no_grad():
            if i < 8: 
                probe_optimizer.param_groups[0]['lr'] = probe_grads_target/(torch.mean(torch.abs(probes_cumulative_grad/num_scans))+torch.std(probes_cumulative_grad/num_scans)) 
        
        probes_param.data[torch.isnan(probes_param)] = 0

    with torch.no_grad():    
            for n3 in range((probes_param.size(2))):    
                    probes_param[:,:,n3].data *= torch.clamp(make_prop_mask(probes_param[:,:,0].data,95,3,psr),1e-8,1)
def apply_LR_scheduler(i,divergence_count_start,loss_tracker,divergence_thr,grads_target,probe_grads_target,optimizer,optimizer_adam,probe_optimizer,tvt_alpha):
    if i > divergence_count_start:            
        if (np.mean(loss_tracker,1)[i-1]) < (np.mean(loss_tracker,1)[i]):
            divergence_count += 1 
            print("loss increasing, count:",divergence_count)

        if divergence_count > divergence_thr:
            grads_target *= 0.5
            probe_grads_target *= 0.5
            optimizer.param_groups[0]['lr'] *= 0.5
            optimizer_adam.param_groups[0]['lr'] *= 0.5
            probe_optimizer.param_groups[0]['lr'] *= 0.5
            
            tvt_alpha *= 0.7


            print('loss increasing, reducing lr, tv')
            divergence_count = 0

def print_metrics(hz_shift,hx_shift,i,iii,num_iters,loss_tracker,allocated_memory,mstime2,mstime1,backtime2,backtime1,GT,out1,tvx,tvy,tvz,tv_total,optimizer,probe_optimizer):
    hz_shift_formatted = " ".join("%2.2E" % x.item() for x in hz_shift.data)
    hx_shift_formatted = " ".join("%2.2E" % x.item() for x in hx_shift.data)
    print()
    print("=======")
    print(i+1, "/", num_iters, "%2.5E" % np.mean(loss_tracker[i]), 
          "mem used:", 
          allocated_memory // (1024 * 1024), "MB",
          "MS fwd time", "%2.2F" % (mstime2-mstime1),
          "Bkwd time", "%2.2F" % ((backtime2-backtime1)))
    print("typical loss", "%2.2E" %torch.mean(torch.abs(torch.abs(GT[:,:,iii])
                                    - torch.abs(out1))**2),
          "typical tvx", "%2.2E" % tvx,
          "typical tvy", "%2.2E" % tvy,
          "typical tvz", "%2.2E" % tvz,
          "typical tv_total", "%2.2E" % tv_total,
          "calculated learning rate", "%2.2E" % (optimizer.param_groups[0]['lr']),
          "calculated probe learning rate", "%2.2E" % (probe_optimizer.param_groups[0]['lr'])
          
         )
    print("=======")
    
    


