import torch
import numpy as np
def wave_interact_partial_voxels(wave, c, slab_thickness, wavelength,ndelta,nbeta):
    return wave * (c*torch.exp(1j * 2 * torch.pi * (ndelta+1j*nbeta) * slab_thickness / wavelength) + (1-c)*torch.exp(1j * 2 * torch.pi * (0+1j*0) * slab_thickness / wavelength))

def wave_interact_partial_voxels_AuAg(wave, c, slab_thickness, wavelength,ndelta,nbeta):
    wavetop,wavesub = torch.split(wave,[full_sim_size[0]-substrate_layers,substrate_layers])
    ctop,csub = torch.split(c,[full_sim_size[0]-substrate_layers,substrate_layers])
    top = wavetop* (ctop*torch.exp(1j * 2 * torch.pi * (ndelta+1j*nbeta) * slab_thickness / wavelength) + (1-ctop)*torch.exp(1j * 2 * torch.pi * (0+1j*0) * slab_thickness / wavelength))
    substrate = wavesub * (csub*torch.exp(1j * 2 * torch.pi * (Audelta+1j*Aubeta) * slab_thickness / wavelength))
    return torch.cat((top,substrate),dim=0)

def wave_interact_full_complex(wave, c, slab_thickness, wavelength):
    return wave * torch.exp(1j * 2 * torch.pi * (c) * slab_thickness / wavelength) 

def wave_interact_AndProp_partial_voxels_deltabeta(wave, c_r, c_i, slab_thickness, wavelength,ndelta,nbeta,tf):
    wave = torch.fft.ifft2(torch.fft.fft2(wave*torch.exp(1j * 2 * torch.pi * 
            (ndelta*(c_r)+1j*nbeta*c_i) * slab_thickness / wavelength))*torch.exp(tf)) 
    return wave

def optimise_probe_prop(probe_in,propdist,lambda_val,voxel_size):
    h1 = get_tf_longprop(propdist,lambda_val,voxel_size,probe_in.size())
    propped_probe_optimised = longprop(probe_in,h1)

    return propped_probe_optimised

def create_freq_mesh(voxel_size,shape):
    u = torch.fft.fftfreq(shape[1])
    v = torch.fft.fftfreq(shape[0])
    vv,uu = torch.meshgrid(v,u)
    vv = vv/voxel_size[0]
    uu = uu/voxel_size[1]
    return uu,vv
# uutest,_ = create_freq_mesh(voxel_size,full_sim_size)
# plt.imshow(uutest.cpu())
def get_tf_longprop(propdist,lambda_val,voxel_size,grid_shape):
    u,v = create_freq_mesh(voxel_size,grid_shape)
    H = torch.exp(-1 * 1j*torch.pi*lambda_val*propdist*(u**2+v**2))
    return H

def get_tf_nearfield(propdist,lambda_val,voxel_size,grid_shape):
    u,v = create_freq_mesh(voxel_size,grid_shape)
    quad = 1-(u**2+v**2)*(lambda_val**2)
    quad_inner = torch.clamp(quad,min=0)
    quad_mask = quad>0
    H = (2j * torch.pi * (propdist / lambda_val)*torch.sqrt(quad_inner))
    
    return H * quad_mask

def farfield_PSI_prop(wave_in,lambda_val,propdist,voxel_size):
    N = wave_in.size()
    g1 = torch.arange(-(N[0]/2),(np.floor((N[0]-1)/2)))
    g2 = torch.arange(-(N[1]/2),(np.floor((N[1]-1)/2)))
    [x,y] = torch.meshgrid(g1,g2)
    r2 = x**2+y**2
    propdist = propdist/voxel_size[0]
    lambda_val = lambda_val/voxel_size[0]
    wout = -1j * torch.exp(1j * torch.pi * lambda_val * propdist * r2 / (N[0]*N[1])) * torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(wave_in * torch.exp(1j * torch.pi * r2 / (lambda_val*propdist)))))
    return wout

def farfield_PSI_prop_2(wave_in,lambda_val,propdist,voxel_size):
    N = wave_in.size()
    g1 = torch.arange(-(N[0]/2),(np.floor((N[0]-1)/2)))
    g2 = torch.arange(-(N[1]/2),(np.floor((N[1]-1)/2)))
    [x,y] = torch.meshgrid(g1,g2)
    r2 = x**2+y**2
    u,v = create_freq_mesh(voxel_size,N)
    u = torch.fft.fftshift(u)/(2*torch.max(u))
    v = torch.fft.fftshift(v)/(2*torch.max(v))
#     propdist = propdist/voxel_size[0]
#     lambda_val = lambda_val/voxel_size[0]

    H = torch.exp(1j * torch.pi * lambda_val/voxel_size[0] * propdist/voxel_size[0] * r2/(N[0]*N[1]))
    pre_exp = -1j * H#torch.exp(1j * torch.pi * lambda_val * propdist * r2 / (N[0]*N[1]))
    tf_inner = torch.exp(1j * torch.pi * r2 / ((lambda_val/voxel_size[0])*(propdist/voxel_size[0])))
    tf_inner_x = torch.exp(1j * torch.pi * (x**2) / ((lambda_val/voxel_size[1])*(propdist/voxel_size[1])))
    tf_inner_y = torch.fft.fftshift(torch.exp(1j * torch.pi * (y**2) / ((lambda_val/voxel_size[0])*(propdist/voxel_size[0]))))
    wout = pre_exp * torch.fft.ifftshift(torch.fft.fft(torch.fft.fft(torch.fft.fftshift(wave_in*tf_inner_x),dim=0)*tf_inner_y,dim=1))
    return wout

def farfield_PSI_prop1d(wave_in,lambda_val,propdist,voxel_size):
    N = wave_in.size()
    g1 = torch.arange(-(N[0]/2),(np.floor((N[0]-1)/2)))
    g2 = torch.arange(-(N[1]/2),(np.floor((N[1]-1)/2)))
    [x,y] = torch.meshgrid(g1,g2)
    r2 = x**2#+y**2
    propdist = propdist/voxel_size
    lambda_val = lambda_val/voxel_size
    wout = -1j * torch.exp(1j * torch.pi * lambda_val * propdist * r2 / (N[0]**2)) * torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(wave_in * torch.exp(1j * torch.pi * r2 / (lambda_val*propdist)))))
    return wout


def longprop(wave_in,h):

    f1 = torch.fft.fft2(wave_in)
    oldflux = torch.sum(torch.abs(f1))
    fh = (f1*h)
    newflux = torch.sum(torch.abs(fh))
    fluxratio = oldflux/newflux

    return torch.fft.ifft2(fh)

def wave_propagate_2d_RI(wavein,h):
    f1 = torch.fft.fft2(wavein)
    oldflux = torch.sum(torch.abs(f1))
    h_exp = (torch.exp(h))
#     h_real = torch.real(h_exp)
#     h_imag = torch.imag(h_exp)
    f1_real = torch.real(f1)
    f1_imag = torch.imag(f1)
    fh_real = f1_real * h_exp - f1_imag * h_exp
    fh_imag = f1_real * h_exp + f1_imag * h_exp
    fh = (fh_real)+1j*(fh_imag)
    newflux = torch.sum(torch.abs(fh))
    fluxratio = oldflux/newflux
    return torch.fft.ifft2((fh*fluxratio))

def fresnel_exit(wave_in,lambda_val,distance,dx,dy):
    
    k = 2*torch.pi/lambda_val
    nx,ny = wave_in.shape
    x = torch.linspace(-nx//2, nx//2 - 1, nx) * dx
    y = torch.linspace(-ny//2, ny//2 - 1, ny) * dy
    X, Y = torch.meshgrid(x, y, indexing='ij')
    fx = torch.fft.fftfreq(nx, d=dx)
    fy = torch.fft.fftfreq(ny, d=dy)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    input_wave_fft = torch.fft.fft2(wave_in)

      # Compute the Fresnel propagation phase term in spatial domain
    phase_term = torch.exp(1j * k * distance) * torch.exp(-1j * k / (2 * distance) * (X**2 + Y**2))
    # Apply the phase term in the Fourier domain
    output_wave_fft = input_wave_fft * phase_term
    output_wave = (output_wave_fft)

    
    return output_wave,phase_term
    


