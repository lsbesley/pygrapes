import torch
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import h5py
def load_scan_positions():
    
    np.random.seed(2)
    file_paths = [r'blank']
    
    all_px_values = []
    all_pz_values = []
    all_ROIlist = []
    all_ROI_inds = []
    full_scan_identifier_list = []
    # Loop over the list of file paths
    file_iter = 0
    n_sub_scans = 1 #load this number of sub scans iwthin a scan
    dataset_shape = np.zeros(len(file_paths)*n_sub_scans)
    all_pz_values = np.load("demo_pz_values.npy")
    all_px_values = np.load("demo_px_values.npy")
    for file_path in file_paths:
    
                full_scan_identifier_list += [file_iter]*all_pz_values.shape[1]
                
                print("subscan",file_iter, "loaded")
                file_iter = file_iter + 1
                
    
    
    
    
    # all_pz_values *= 0.6
    # Calculate ROIlist as in your code
    px_block = [5,25] 
    ROI_inds = np.where((all_px_values < px_block[1]) & (all_px_values > px_block[0]) & (all_pz_values < 105) & (all_pz_values > 25) | #Keep at 65, 35!!
                        (all_px_values < px_block[1]) & (all_px_values > px_block[0]) & (all_pz_values < 0) & (all_pz_values > 0) |
                       
                        (all_px_values < px_block[1]) & (all_px_values > px_block[0]) & (all_pz_values < 0) & (all_pz_values > 0)) 
    
    cvals = np.zeros((1,all_px_values.shape[1]))
    
    cvals[0][ROI_inds[1]] = 1
    print("size of scans in ROI:",all_px_values[ROI_inds].size)
    
    scan_categories = np.cumsum(dataset_shape)
    print("scan_categories",scan_categories)
    
    #which subset is a list where each value is each file it belongs to.
    which_subset = np.digitize(ROI_inds[1],scan_categories)
    
    #this corrects the scan number by subtracging the previous number of scans from ecah scan, since each subscan starts from zero... 
    #this is to correctly load the h5 with teh correct indices.
    scan_no_corrector = np.zeros_like(which_subset)
    
    sc2 = np.insert(scan_categories,0,0)[:-1]
    # for n1 in range(1,(which_subset.shape[0])):
    #     scan_no_corrector[n1] = sc2[which_subset[n1]]#-sc2[which_subset[n1]]
    #some bug?
    scan_no_corrector[0] = scan_no_corrector[1]
    
    corrected_ROI_inds = (ROI_inds[1]-scan_no_corrector)
    
    ROI_inds_sub = [[] for _ in range(len(file_paths)*n_sub_scans)]
    for ii in range(len(file_paths)*n_sub_scans):
        ROI_inds_sub[ii] = corrected_ROI_inds[which_subset==(ii)].astype(int).tolist()
    
    num_scans = all_px_values[ROI_inds].size
    
    
    scan_inc_angle_index = [0.7]
    for ii in range(len(ROI_inds_sub)):
        
        if ii == 0:
            inc_angle_list = np.ones(len(ROI_inds[1]))*scan_inc_angle_index[ii]
            
            scan_identifier_list = np.ones(len(ROI_inds[1]))*ii
        else:
            inc_angle_list = np.concatenate((inc_angle_list,np.ones(len(ROI_inds_sub[ii]))*scan_inc_angle_index[ii]))
            scan_identifier_list = np.concatenate((scan_identifier_list,np.ones(len(ROI_inds_sub[ii]))*ii))
    
    #fix inc angles list 
    
    return inc_angle_list,scan_identifier_list,num_scans,ROI_inds_sub,corrected_ROI_inds,ROI_inds,cvals,all_px_values,all_pz_values,full_scan_identifier_list

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

