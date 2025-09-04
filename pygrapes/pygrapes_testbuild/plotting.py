import torch
import numpy as np
import matplotlib.pyplot as plt
from .probe_aux import combine_probe_modes

def plot_scan_points(cvals,full_scan_identifier_list,all_px_values,all_pz_values):
    colors = []
    for val in cvals[0]:
        
        if val == 1:
            colors.append('red')
        else: 
            colors.append('blue')
    markers_list = ['x' if value == 0 else 'x' for value in full_scan_identifier_list]
    # Define a mapping from scan identifiers to color transformations
    color_table = {
        0: {'blue': 'mediumblue', 'red': 'cornflowerblue'},
        1: {'blue': 'mediumblue', 'red': 'lightsteelblue'},
        2: {'blue': 'mediumblue', 'red': 'fuchsia'},
        3: {'blue': 'darkviolet', 'red': 'deeppink'},
        4: {'blue': 'darkviolet', 'red': 'hotpink'},
        5: {'blue': 'darkviolet', 'red': 'gold'},
        7: {'blue': 'darkred', 'red': 'tomato'},
        7: {'blue': 'darkred', 'red': 'orangered'},
        8: {'blue': 'darkred', 'red': 'coral'},
        9: {'blue': 'darkolivegreen', 'red': 'springgreen'},
        10: {'blue': 'darkolivegreen', 'red': 'limegreen'},
        11: {'blue': 'darkolivegreen', 'red': 'palegreen'},
    }
    
    # Iterate through each index and update colors based on full_scan_identifier_list
    for ii in range(len(full_scan_identifier_list)):
        scan_id = full_scan_identifier_list[ii]
        if scan_id in color_table:
            current_color = colors[ii]
            if current_color in color_table[scan_id]:
                colors[ii] = color_table[scan_id][current_color]
    
    plt.figure(figsize=[15,10])
    for apx, apz, color,marks in zip(all_px_values[0], all_pz_values[0], colors,markers_list):
        plt.scatter(apx, apz, c=color, s=70,marker=marks)


def plot_reconstruction(fig, xr, iii, probes_param, out1, this_GT,
                        dzz, dzy, yzpad, use_multiple_probe_modes,
                        probe_vmax, recon_FFT_vmin, recon_FFT_vmax,
                        psr, i, flux_ratio,scan_identifier_list):
    
    if i > 0:
        plt.close(fig)
        plt.close()
        fig = plt.figure(figsize=[10, 10])

    ax1 = fig.add_subplot(2, 2, 1)
    if not ((dzz == 0) & (dzy == 0) & (yzpad == 0)):
        im1 = ax1.imshow(
            torch.sum(xr[:, (dzy+yzpad):-(dzy+yzpad), (dzz+yzpad):-(dzz+yzpad)].detach().cpu(), dim=0),
            interpolation='none', aspect=2.5)
    else:
        im1 = ax1.imshow(xr[0, :, :].detach().cpu(), interpolation='none', aspect=1)
    fig.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(2, 2, 2)
    probe_image = torch.abs(combine_probe_modes(probes_param, iii,scan_identifier_list)).detach().cpu()
    im2 = ax2.imshow(probe_image, vmin=0, vmax=probe_vmax, interpolation='none', aspect=1/psr)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(torch.log((torch.abs(out1) * flux_ratio) + 1e-9).detach().cpu(),
               vmin=recon_FFT_vmin, vmax=recon_FFT_vmax, interpolation='None')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(torch.log(torch.abs(this_GT) + 1e-9).detach().cpu(),
               vmin=recon_FFT_vmin, vmax=recon_FFT_vmax, interpolation='None')

    display(fig)
    return fig


