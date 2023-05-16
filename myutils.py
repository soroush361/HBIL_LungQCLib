import os
import shutil as sh
import glob
import json
import logging
from tqdm.notebook import tqdm
from multiprocessing import Pool
from datetime import datetime
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import nibabel as ni
import cupy as cp
from scipy.ndimage import uniform_filter
from scipy.io import loadmat
import numbers
from cupy.lib.stride_tricks import as_strided
from functools import partial

import Jie_PDCM

    
# All pipeline functions
###################################################################################### End of class
class PLF:
    # Make a subject case
    def get_sub(Cohort, Study, ID, PID, DIR, CT_path, Lung_path, Airway_path, sLTP_path='',
                FullName=None, Scanner='Unknown', Resolution=None, LungRange=None, ULN=None, SMOKE_PACK_YEARS=None, CURRENT_SMOKER=None, Alpha1Status=None
               ):

        scan = ni.load(CT_path)
        shape = scan.shape
        Res = tuple(np.abs(np.array(scan.header['pixdim'][1:4])).astype(float)) if Resolution is None else Resolution

        sub = dict()
        sub['Cohort'] = Cohort
        sub['Study'] = Study
        sub['ID'] = ID
        sub['PID'] = PID
        sub['DIR'] = DIR
        sub['Originals'] = dict()
        sub['Originals']['CT'] = CT_path
        sub['Originals']['Lung'] = Lung_path
        sub['Originals']['Airway'] = Airway_path
        sub['Originals']['sLTP'] = sLTP_path

        sub['FullName'] = os.path.basename(CT_path).replace('.gz', '').replace('.nii', '').replace('_IMG', '') if FullName is None else FullName
        sub['Scanner'] = Scanner
        sub['Shape'] = shape
        sub['Resolution'] = Res
        sub['LungRange'] = dict({'X':[0,0],'Y':[0,0],'Z':[0,0]}) if LungRange is None else LungRange

        sub['CT'] = f'{DIR}/{ID}_CT.nii.gz'
        sub['Lung'] = f'{DIR}/{ID}_Lung.nii.gz'
        sub['Airway'] = f'{DIR}/{ID}_Airway.nii.gz'
        sub['R'] = f'{DIR}/{ID}_R.nii.gz'
        sub['Theta'] = f'{DIR}/{ID}_Theta.nii.gz'
        sub['Phi'] = f'{DIR}/{ID}_Phi.nii.gz'
        sub['SpatialIndex'] = f'{DIR}/{ID}_SPIdx.nii.gz'
        sub['LungVolume'] = f'{DIR}/{ID}_LungVol.nii.gz'
        sub['950Volume'] = f'{DIR}/{ID}_950Vol.nii.gz'
        sub['PercentEmph'] = f'{DIR}/{ID}_PercentEmph.nii.gz'
        sub['HMMF'] = f'{DIR}/{ID}_HMMF.nii.gz'
        sub['T3D'] = f'{DIR}/{ID}_T3D.nii.gz'
        sub['sLTP'] = f'{DIR}/{ID}_sLTP.nii.gz'
        sub['sLTP36'] = f'{DIR}/{ID}_sLTP36.nii.gz'
        sub['ULN_Mask'] = f'{DIR}/{ID}_ULNMask.nii.gz'
        sub['ULN_sLTP'] = f'{DIR}/{ID}_ULNsLTP.nii.gz'

        sub['ULN'] = ULN
        sub['SMOKE_PACK_YEARS'] = SMOKE_PACK_YEARS
        sub['CURRENT_SMOKER'] = CURRENT_SMOKER
        sub['Alpha1Status'] = Alpha1Status

        return sub
    
    def normalize_CT(CT):
        CT = np.clip(CT, -1000, -400)
        CT = (CT+1000)/600
        return CT

    def can_crop(x,y,z, img_shape, ROI_size):
        point = np.array([x,y,z])
        lower = ROI_size // 2
        upper = ROI_size // 2 + ROI_size % 2

        return not (np.any(point - lower < 0) or np.any(point+upper >= np.array(img_shape)))
    
    def crop(x,y,z, img, ROI_size):
        point = np.array([x,y,z])
        lower = tuple((point - (ROI_size // 2)).astype('int'))
        upper = tuple((point + (ROI_size // 2 + ROI_size % 2)).astype('int'))
        return img[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    
    def update_sub_dir(sub, DIR):
        origDIR = sub['DIR']
        sub['DIR'] = DIR
        for key in sub:
            if isinstance(sub[key], str) and key != 'DIR' and sub[key].startswith(origDIR):
                sub[key] = sub[key].replace(origDIR, DIR)
        return sub
    
    def load_subjects(file_path):
        with open(file_path, 'r') as f:
            subjects = json.load(f)
        return subjects

    def save_subjects(file_path, subjects):
        with open(file_path, 'w') as f:
            json.dump(subjects, f)

    def write_subjects_info(subjects):
        for sub in subjects:
            PLF.write_sub_info(sub)
    
    def write_sub_info(sub):
        DIR = sub['DIR']
        ID = sub['ID']
        with open(f'{DIR}/{ID}_Info.json', 'w') as f:
            json.dump(sub, f)

    def read_subjects_info(study_path):
        info_files = sorted(glob.glob(f'{study_path}/*/*_Info.json'))
        subjects = list()
        for inf in info_files:
            subjects.append(PLF.read_sub_info(inf))
                
    def read_sub_info(sub_file):
        with open(sub_file, 'r') as f:
            sub = json.load(f)
        return sub
    
    def get_sub_byID(ID, subjects):
        subs = [s for s in subjects if s['ID'] == ID]
        if len(subs) == 0:
            raise Exception(f'Subject ID {ID} does not exist in subjects.')
        elif len(subs) > 1:
            raise Exception(f'Multiple subjects with ID = {ID} found in subjects.')
        return subs[0]
    
    # Make directories
    def make_sub_directories(sub):
        DIR = sub['DIR']
        if not os.path.isdir(f'{DIR}'):
            os.mkdir(f'{DIR}')
        if not os.path.isdir(f'{DIR}/logs'):
            os.mkdir(f'{DIR}/logs')
        if not os.path.isdir(f'{DIR}/tmp'):
            os.mkdir(f'{DIR}/tmp')
        if not os.path.isdir(f'{DIR}/QC_Figures'):
            os.mkdir(f'{DIR}/QC_Figures')
        if not os.path.isdir(f'{DIR}/ROIs'):
            os.mkdir(f'{DIR}/ROIs')
    
    # make directory and get the logger
    def get_logger(DIR, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        handler = logging.FileHandler(f'{DIR}/{name}_{now}.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def close_logger(logger):
        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.close()
        del logger
    
    # Clean logs and tmp
    def clean_logs_tmp(sub, logs=True, tmp=True):
        DIR = sub['DIR']
        if logs:
            if os.path.isdir(f'{DIR}/logs'):
                for f in glob.glob(f'{DIR}/logs/*.txt'):
                    os.remove(f)
        if tmp:
            if os.path.isdir(f'{DIR}/tmp'):
                for f in glob.glob(f'{DIR}/tmp/*'):
                    os.remove(f)
                    
    # Loads images
    def load_image(image_path, dtype=np.float32, returnHeaderInfo=False):
        scan = ni.load(image_path)
        img = scan.get_fdata().astype(dtype)
        if returnHeaderInfo:
            return img, scan.header, scan.affine
        else:
            return img
    
    # Saves images
    def save_image(image_path, img, dtype, header, affine):
        img = ni.Nifti1Image(img.astype(dtype), header=header, affine=affine)
        ni.save(img, image_path)
        
    def make_dtype_header(header, dtype):
        dtype_header = header.copy()
        dtype_header.set_data_dtype(dtype)
        return dtype_header
    
    def make_affine_and_headers(resolution, CT_header):
        affine = np.zeros((4,4))
        affine[:3,:3] = np.diag(resolution)
        affine[3,3] = 1
        
        int_header = PLF.make_dtype_header(CT_header, 'int16')
        float_header = PLF.make_dtype_header(CT_header, 'float32')
        
        return affine, int_header, float_header
    
    # Checks images value to ensure they are in range
    def check_values_minmax(img, expected_minV, expected_maxV):
        minV, maxV = np.min(img), np.max(img)
        return minV >= expected_minV and maxV <= expected_maxV, minV, maxV
    
    def check_values_unique(img, expected_uniqueV):
        expected_uniqueV = np.array(expected_uniqueV)
        uniqueV = np.unique(img)
        if len(expected_uniqueV) != len(uniqueV):
            return False, uniqueV
        return np.all(uniqueV == expected_uniqueV), uniqueV
    
    def find_best_flip(CT, mask):
        flip_axis = list([None, 0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)])
        with_transpose = list([False for _ in range(len(flip_axis))])
        scores = np.zeros((len(flip_axis),))
        for i, ax in enumerate(flip_axis):
            if ax is not None:
                tmpmask = np.flip(mask, axis=ax)
            else:
                tmpmask = mask
            # Without xy transpose
            tmp = CT[(tmpmask > 0)]
            h, _ = np.histogram(tmp, bins=100, range=[-1024, 0])
            score1 = h[50:].sum()/h[:50].sum()

            # With xy transpose
            tmpmask = tmpmask.transpose(1,0,2)
            tmp = CT[(tmpmask > 0)]
            h, _ = np.histogram(tmp, bins=100, range=[-1024, 0])
            score2 = h[50:].sum()/h[:50].sum()

            if score1 < score2:
                scores[i] = score1
                with_transpose[i] = False
            else:
                scores[i] = score2
                with_transpose[i] = True

        del tmp, tmpmask, h
        minIDX = np.argmin(scores)
        return flip_axis[minIDX], with_transpose[minIDX]

    def correct_flips(CT, Lung, AW):
        mess = ''
        # Align Lung and AW
        flip_suggest, with_transpose = PLF.find_best_flip(CT, Lung)
        if flip_suggest is not None:
            Lung = np.flip(Lung, axis=flip_suggest)
            mess += f'Lung mask flipped along {str(flip_suggest)} axis to align with CT.\n'
        if with_transpose:
            Lung = Lung.transpose(1,0,2)
            mess += f'Lung mask transposed along xy axes to align with CT.\n'

        flip_suggest, with_transpose = PLF.find_best_flip(CT, AW)
        if flip_suggest is not None:
            AW = np.flip(AW, axis=flip_suggest)
            mess += f'Airway mask flipped along {str(flip_suggest)} axis to align with CT.\n'
        if with_transpose:
            AW = AW.transpose(1,0,2)
            mess += f'Airway mask transposed along xy axes to align with CT.\n'

        # Check if xy transpose is needed
        diff1 = np.diff(Lung.sum(axis=(0, 2)))
        diff2 = np.diff(Lung.sum(axis=(1, 2)))
        if diff1[:255].sum() + np.abs(diff1[255:].sum()) < diff2[:255].sum() + np.abs(diff2[255:].sum()):
            CT = CT.transpose(1, 0, 2)
            Lung = Lung.transpose(1, 0, 2)
            AW = AW.transpose(1, 0, 2)
            mess += f'All images transposed to ensure x is x and y is y.\n'

        # Find if Right and Left is mixed and correct it
        sums = np.sum(Lung, axis=(1,2))
        if sums[:256].sum() / sums[256:].sum() > 1:
            CT = np.flip(CT, axis=0)
            Lung = np.flip(Lung, axis=0)
            AW = np.flip(AW, axis=0)
            mess += f'All images flipped along 0 axis to ensure correct left and right.\n'

        # Find if anterior and posterior is switched
        sums = np.sum(CT>100, axis=(0,2))
        if sums[:300].sum() / sums[300:].sum() < 1:
            CT = np.flip(CT, axis=1)
            Lung = np.flip(Lung, axis=1)
            AW = np.flip(AW, axis=1)
            mess += f'All images flipped along 1 axis to ensure correct posterior and anterior.\n'

        # Find if supperior is not up
        sums = np.sum(AW, axis=(0,1))
        hf = len(sums) // 2
        if sums[:hf].sum() / sums[hf:].sum() > 1:
            CT = np.flip(CT, axis=2)
            Lung = np.flip(Lung, axis=2)
            AW = np.flip(AW, axis=2)
            mess += f'All images flipped along 2 axis to ensure correct superior and inferior.\n'

        return CT, Lung, AW, mess
    
    # Ensures XY are the first axes and 512 voxels
    def check_XY512(image):
        if len(image.shape) != 3:
            return False
        return image.shape[0] == image.shape[1] and image.shape[0] == 512
    
    # Ensures all images have the same shape as CT
    def check_shapes(expected_shape, images):
        results = list()
        diffs = list()
        for img in images:
            results.append(img.shape == expected_shape)
            diffs.append(tuple(np.array(img.shape) - np.array(expected_shape)))
        return results, diffs
    
    # Extracts lung bounding box
    def extract_lung_bounding_box(sub):
        Lung = PLF.load_image(sub['Lung']) > 0
        x,y,z = np.where(Lung)
        del Lung
        return dict({'X': [int(x.min()), int(x.max())], 'Y': [int(y.min()), int(y.max())], 'Z': [int(z.min()), int(z.max())]})
    
    # A general plot function.
    def plot(images,
             suptitle='', titles=None, save_path='', maxFontSize=25, 
             cmaps=None, minvs=None, maxvs=None, globalMinMax=True,
             showColorbar=True, colorbarTitles=None, isDiscrete=None,
             interpolation='none', dpi=70, 
             show=True, showSlice=True, sliceColor='lime', plotGrid='auto',
             x=None, y=None, z=None):
        
        is_list = isinstance(images, list)
        images = images if is_list else list([images])
        titles = titles if is_list else list([titles])
        cmaps = cmaps if is_list else list([cmaps])
        colorbarTitles = colorbarTitles if is_list else list([colorbarTitles])
        isDiscrete = isDiscrete if is_list else list([isDiscrete])
        minvs = minvs if is_list else list([minvs])
        maxvs = maxvs if is_list else list([maxvs])

        n_imgs = len(images)

        titles = list([None for _ in range(n_imgs)]) if titles is None else titles
        colorbarTitles = list([None for _ in range(n_imgs)]) if colorbarTitles is None else colorbarTitles
        isDiscrete = list([False for _ in range(n_imgs)]) if isDiscrete is None else isDiscrete
        cmaps = list(['jet' for _ in range(n_imgs)]) if cmaps is None else cmaps
        if len(cmaps) == 1 and cmaps[0] is None:
            cmaps[0] = 'jet'

        minvs = list([None for _ in range(n_imgs)]) if minvs is None else minvs
        maxvs = list([None for _ in range(n_imgs)]) if maxvs is None else maxvs

        n_row = None
        if isinstance(plotGrid, str):
            if plotGrid.lower() != 'auto':
                raise Exception(f'Plot grid of {str(plotGrid)} is not supported.')
            if n_imgs <= 4:
                im_n_col = 1
            elif n_imgs <= 8:
                im_n_col = 2
            else:
                im_n_col = 3
        else:
            n_row = plotGrid[0]
            im_n_col = plotGrid[1]

        n_col = im_n_col * 3
        n_row = int(np.ceil(n_imgs / im_n_col)) if n_row is None else n_row

        shape = images[0].shape    
        x_slc = shape[0] // 3 if x is None else x
        y_slc = shape[1] // 2 if y is None else y
        z_slc = shape[2] // 2 if z is None else z

        if globalMinMax:
            minvs = list([(images[i].min() if minv is None else minv) for i, minv in enumerate(minvs)])
            maxvs = list([(images[i].max() if maxv is None else maxv) for i, maxv in enumerate(maxvs)])
        else:
            minvs = list([(np.min((images[i][:,:,z_slc].min(), images[i][:,y_slc,:].min(), images[i][x_slc,:,:].min())) if minv is None else minv) for i, minv in enumerate(minvs)])
            maxvs = list([(np.max((images[i][:,:,z_slc].max(), images[i][:,y_slc,:].max(), images[i][x_slc,:,:].max())) if maxv is None else maxv) for i, maxv in enumerate(maxvs)])

        fig, axes = plt.subplots(n_row, n_col, figsize=(12 + n_col*2, 5+n_row*3), dpi=dpi)
        if len(axes.shape) <= 1:
            axes = np.reshape(axes, (1, len(axes)))
        if suptitle != '':
            fig.suptitle(suptitle, fontsize=maxFontSize)
        for i, (img, minv, maxv, cmap_n, title, cbarTitle, isDesc) in enumerate(zip(images, minvs, maxvs, cmaps, titles, colorbarTitles, isDiscrete)):
            if img is None:
                continue

            cmap = cm.get_cmap(cmap_n)
            if isDesc:
                cbar_lbls = np.arange(minv, maxv+1).astype('int')
                cmap = cm.get_cmap(cmap_n, len(cbar_lbls))
                cbar_ticks = (cbar_lbls+0.5)*(maxv-minv)/len(cbar_lbls)
                if len(cbar_lbls) > 15:
                    cbar_lbls = cbar_lbls.astype(str)
                    for ii in range(1, len(cbar_lbls)-1, 5):
                        cbar_lbls[ii:ii+4] = ''
                        if ii+4 == len(cbar_lbls) - 2:
                            cbar_lbls[ii+4] = ''
                            
            # Axial
            ax = axes[i*3//n_col, (i*3)%n_col+0]
            if title is not None:
                ax.set_title(title, fontsize=max(maxFontSize-5, 5))
            im = ax.imshow(np.rot90(img[:,:,z_slc]), cmap=cmap, vmin=minv, vmax=maxv, interpolation=interpolation)
            if showSlice:
                ax.axhline(shape[1] - y_slc, lw=1, c=sliceColor)
                ax.axvline(x_slc, lw=1, c=sliceColor)  
            if showColorbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("bottom", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
                if isDesc:
                    cbar.ax.set_xticks(cbar_ticks)
                    cbar.ax.set_xticklabels(cbar_lbls)
                if cbarTitle is not None:
                    cbar.set_label(cbarTitle, fontsize=max(maxFontSize-15, 5))
                ax.get_xaxis().set_visible(False)

            ticks = ax.get_yticks()[1:-1]
            ax.set_yticks(shape[1] - ticks[::-1])
            ax.set_yticklabels(ticks[::-1].astype('int'))

            # Sagittal
            ax = axes[i*3//n_col, (i*3)%n_col+1]
            im = ax.imshow(np.rot90(img[:,y_slc,:]), cmap=cmap, vmin=minv, vmax=maxv, interpolation=interpolation)
            if showSlice:
                ax.axhline(shape[2] - z_slc, lw=1, c=sliceColor)
                ax.axvline(x_slc, lw=1, c=sliceColor)

            ticks = ax.get_yticks()[1:-1]
            ax.set_yticks(shape[2] - ticks[::-1])
            ax.set_yticklabels(ticks[::-1].astype('int'))

            # Choronal
            ax = axes[i*3//n_col, (i*3)%n_col+2]
            im = ax.imshow(np.rot90(img[x_slc,:,:]), cmap=cmap, vmin=minv, vmax=maxv, interpolation=interpolation)
            if showSlice:
                ax.axhline(shape[2] - z_slc, lw=1, c=sliceColor)
                ax.axvline(y_slc, lw=1, c=sliceColor)

            ticks = ax.get_yticks()[1:-1]
            ax.set_yticks(shape[2] - ticks[::-1])
            ax.set_yticklabels(ticks[::-1].astype('int'))

        plt.tight_layout()

        if save_path != '':
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
            
            
    # Take an screenshot of everything!
    def plot_screenshots(sub, show=True, returnImages=False):
        Cohort = sub['Cohort']
        Study = sub['Study']
        ID = sub['ID']
        DIR = sub['DIR']

        spl_diam = 25 # mm
        resolution = np.array(sub['Resolution'])
        window_size = tuple(np.ceil(spl_diam/resolution).astype('int'))

        CT = PLF.load_image(sub['CT'])
        Lung = PLF.load_image(sub['Lung'])
        Airway = PLF.load_image(sub['Airway'])
        R = PLF.load_image(sub['R'])
        Theta = PLF.load_image(sub['Theta'])
        Phi = PLF.load_image(sub['Phi'])
        SpatialIndex = PLF.load_image(sub['SpatialIndex'])
        LungVolume = PLF.load_image(sub['LungVolume'])
        PercentEmph = PLF.load_image(sub['PercentEmph'])
        T3D = PLF.load_image(sub['T3D'])
        sLTP36 = PLF.load_image(sub['sLTP36'])
        sLTP = PLF.load_image(sub['sLTP'])
        Emph_Mask = PLF.load_image(sub['ULN_Mask'])
        F_sLTP = PLF.load_image(sub['ULN_sLTP'])
        
        if os.path.isfile(sub['Originals']['sLTP']):
            Jie_sLTP = PLF.load_image(sub['Originals']['sLTP'])
            Jie_sLTP = np.flip((11 - Jie_sLTP), axis=-1)
            Jie_sLTP[Jie_sLTP > 10] = 0
        else:
            Jie_sLTP = None
            
            
        PLF.plot([CT, Lung, Airway, R, Theta, Phi, SpatialIndex, LungVolume, PercentEmph, T3D, sLTP36, sLTP, Emph_Mask, F_sLTP, Jie_sLTP], 
                 suptitle=f'{Cohort} - {Study} - {ID} - All Preprocessing',
                 titles=['CT', 'Lung', 'Airway', 'R', 'Theta', 'Phi', 'Spatial Index', 'Lung Volume', '%emph-950', 'T3D', 'sLTP (Original)', 'sLTP (Corrected)', 'Emphysema mask', 'Final sLTP', 'Jie\'s sLTP'],
                 save_path=f'{DIR}/QC_Figures/{ID}_FullScreenShots.png',
                 globalMinMax=False,
                 minvs=[-1024, 0, 0, 0, -np.pi, -np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 maxvs=[200, 30, 1, 1, np.pi, np.pi/2, 36, np.prod(window_size), None, 40, 10, 10, 1, 10, 10],
                 isDiscrete=[False, True, True, False, False, False, True, False, False, True, True, True, True, True, True],
                 colorbarTitles=['HU', None, None, 'Distance', 'Radian', 'Radian', 'Spatial Index', 'Voxels', '%', 'Texton label', 'sLTP label', 'sLTP label', None, 'sLTP label', 'sLTP label'],
                 cmaps=['gray', 'jet', 'gray', 'jet', 'jet', 'jet', 'jet', 'jet', 'jet', 'jet', 'jet', 'jet', 'gray', 'jet', 'jet'], show=show
                )
        if returnImages:
            return CT, Lung, Airway, R, Theta, Phi, SpatialIndex, LungVolume, PercentEmph, T3D, sLTP36, sLTP, Emph_Mask, F_sLTP, Jie_sLTP
        else:
            del CT, Lung, Airway, R, Theta, Phi, SpatialIndex, LungVolume, PercentEmph, T3D, sLTP36, sLTP, Emph_Mask, F_sLTP, Jie_sLTP
        
        
        
        
    
    # QC and save corrected inputs
    def QC_inputs(sub, overwrite=False):
        try:
            Cohort = sub['Cohort']
            Study = sub['Study']
            ID = sub['ID']
            DIR = sub['DIR']
            resolution = sub['Resolution']
            logger = PLF.get_logger(f'{DIR}/logs', f'{ID}_InputQC')
            logger.info(f'Input images QC and correction started for subject {ID} of cohort {Cohort} - {Study}.')

            if not overwrite:
                should_continue = True
                for img_key in ['CT', 'Lung', 'Airway']:
                    if os.path.isfile(sub[img_key]):
                        logger.error(f'Output directory contains {img_key} image in {sub[img_key]}. Please remove then re-run QC.')
                        should_continue = False
                if not should_continue:
                    PLF.close_logger(logger)
                    return False

            CT_path = sub['Originals']['CT']
            Lung_path = sub['Originals']['Lung']
            AW_path = sub['Originals']['Airway']

            CT, header, _ = PLF.load_image(CT_path, dtype=np.int16, returnHeaderInfo=True)
            Lung = PLF.load_image(Lung_path, dtype=np.int16)
            AW = PLF.load_image(AW_path, dtype=np.int16)
            
            should_continue = True
            if not PLF.check_XY512(CT):
                logger.error(f'CT image must be a 512x512xSlices shape. Current shape is {CT.shape}.')
                should_continue = False
            if not PLF.check_XY512(Lung):
                logger.error(f'Lung mask must be a 512x512xSlices shape. Current shape is {Lung.shape}.')
                should_continue = False
            if not PLF.check_XY512(AW):
                logger.error(f'Airway mask must be a 512x512xSlices shape. Current shape is {AW.shape}.')
                should_continue = False
                
            res, diff = PLF.check_shapes(CT.shape, [CT, Lung, AW])
            for i, (r, img, lbl) in enumerate(zip(res, [CT, Lung, AW], ['CT', 'Lung', 'Airway'])):
                if not r:
                    should_continue = False
                    logger.error(f'Expected array shape is {CT.shape}, but {lbl} image has a shape of {img.shape}. (Diff = {diff[i]})')
            if not should_continue:
                PLF.close_logger(logger)
                return False

            affine, int_header, float_header = PLF.make_affine_and_headers(resolution, header)

            is_inrange, minV, maxV = PLF.check_values_minmax(CT, expected_minV=-5000, expected_maxV=5000)
            if not is_inrange:
                logger.warning(f'CT image is expected to have values between [-5000, 5000] HU, but the CT has [{minV}, {maxV}] HU values.')

            is_inrange, uniqueV = PLF.check_values_unique(Lung, expected_uniqueV=[0, 20, 30])
            if not is_inrange:
                logger.warning(f'Lung image is expected to have values [0, 20, 30], but the Lung mask has [{uniqueV}] values.')

            is_inrange, uniqueV = PLF.check_values_unique(AW, expected_uniqueV=[0, 1])
            if not is_inrange:
                logger.warning(f'Airway image is expected to have values [0, 1], but the Airway mask has [{uniqueV}] values. Anything larger than 0 will be considered as Airway.')
                AW = (AW > 0).astype('int16')

            PLF.plot([CT, Lung, AW], 
                     suptitle=f'{Cohort} - {Study} - {ID} - Original input scans',
                     save_path=f'{DIR}/QC_Figures/{ID}_OriginalInputs.png',
                     titles=['CT', 'Lung mask', 'Airway mask'],
                     cmaps=['gray', 'jet', 'gray'],
                     minvs=[-1024, 0, 0],
                     maxvs=[200, 30, 1],
                     isDiscrete=[False, True, True],
                     colorbarTitles=['HU', None, None],
                     show=False,
                    )
            
            logger.info(f'Images are loaded from:\nCT: {CT_path}\nLung: {Lung_path}\nAirway: {AW_path}')

            CT, Lung, AW, mess = PLF.correct_flips(CT, Lung, AW)
            if mess != '':
                logger.warning(mess)

            PLF.save_image(sub['CT'], CT, np.int16, int_header, affine)
            PLF.save_image(sub['Lung'], Lung, np.int16, int_header, affine)
            PLF.save_image(sub['Airway'], AW, np.int16, int_header, affine)
        
            
            PLF.plot([CT, Lung, AW], 
                     suptitle=f'{Cohort} - {Study} - {ID} - Processed input scans',
                     save_path=f'{DIR}/QC_Figures/{ID}_ProcessedInputs.png',
                     titles=['CT', 'Lung mask', 'Airway mask'],
                     cmaps=['gray', 'jet', 'gray'],
                     minvs=[-1024, 0, 0],
                     maxvs=[200, 30, 1],
                     isDiscrete=[False, True, True],
                     colorbarTitles=['HU', None, None],
                     show=False,
                    )

            logger.info(f'CT, Lung, and Airway images were corrected and saved. Screen shots are availabel in {DIR}/QC_Figures.')
            logger.info(f'Process is complete now.')
            PLF.close_logger(logger)
            return True
        except Exception as e:
            PLF.close_logger(logger)
            raise e
            
    def Redraw_QC_Plot(sub, overwrite=False):
        try:
            Cohort = sub['Cohort']
            Study = sub['Study']
            ID = sub['ID']
            DIR = sub['DIR']
            logger = PLF.get_logger(f'{DIR}/logs', f'{ID}_RedrawQCShots')
            logger.info(f'Re-drawing QC images started for subject {ID} of cohort {Cohort} - {Study}. (Usually, used for manual corrections)')

            if os.path.isfile(f'{DIR}/QC_Figures/{ID}_ProcessedInputs.png'):
                counter = len(glob.glob(f'{DIR}/QC_Figures/{ID}_ProcessedInputs*.png'))
                logger.info(f'{counter} QC screenshots of previously processed images were found. The last one will be numbered, and the new one will be replace.')
                sh.copyfile(f'{DIR}/QC_Figures/{ID}_ProcessedInputs.png', f'{DIR}/QC_Figures/{ID}_ProcessedInputs_{counter}.png')

            CT = PLF.load_image(sub['CT'], dtype=np.int16)
            Lung = PLF.load_image(sub['Lung'], dtype=np.int16)
            AW = PLF.load_image(sub['Airway'], dtype=np.int16)

            PLF.plot([CT, Lung, AW], 
                         suptitle=f'{Cohort} - {Study} - {ID} - Processed input scans',
                         save_path=f'{DIR}/QC_Figures/{ID}_ProcessedInputs.png',
                         titles=['CT', 'Lung mask', 'Airway mask'],
                         cmaps=['gray', 'jet', 'gray'],
                         minvs=[-1024, 0, 0],
                         maxvs=[200, 30, 1],
                         isDiscrete=[False, True, True],
                         colorbarTitles=['HU', None, None],
                         show=False,
                        )
            logger.info(f'New QC images were created for subject {ID} of cohort {Cohort} - {Study}.')
            PLF.close_logger(logger)
            return True
        except Exception as e:
            PLF.close_logger(logger)
            raise e
    
    # Converts R, Theta, and Phi to SPIdx
    def convert_to_SPIdx(r, t, p):
        # It's not the exact impementation of Jie's, but SP feature do not contibute in sLTP label much! (less than 0.5% observed.z1)
        eps = 1e-10
        r_bins = list([0-eps, 1/3, 2/3, 1]) # 4
        n_r_bins = len(r_bins) - 1
        theta_bins = list([-np.pi-eps, -np.pi/2, 0, np.pi/2, np.pi]) # 5
        n_theta_bins = len(theta_bins) - 1
        phi_bins = list([-np.pi/2-eps, -np.pi/6, np.pi/6, np.pi/2]) # 4
        n_phi_bins = len(phi_bins) - 1

        i = np.digitize(r, r_bins, right=True) - 1
        j = np.digitize(t, theta_bins, right=True) - 1
        k = np.digitize(p, phi_bins, right=True) - 1

        return i*n_theta_bins*n_phi_bins+j*n_phi_bins+k + 1
    
    
    # Generate PDCM
    def Make_PDCM(sub, overwrite=False):
        try:
            Cohort = sub['Cohort']
            Study = sub['Study']
            ID = sub['ID']
            DIR = sub['DIR']
            resolution = sub['Resolution']
            logger = PLF.get_logger(f'{DIR}/logs', f'{ID}_PDCMMaker')
            logger.info(f'Creating PDCM images started for subject {ID} of cohort {Cohort} - {Study}.')

            if not overwrite:
                should_continue = True
                for img_key in ['R', 'Theta', 'Phi', 'SpatialIndex']:
                    if os.path.isfile(sub[img_key]):
                        logger.error(f'Output directory contains {img_key} image in {sub[img_key]}. Please remove then re-run PDCM maker.')
                        should_continue = False
                if not should_continue:
                    PLF.close_logger(logger)
                    return False

            pdcm = Jie_PDCM.initialize()
            pdcm.make_PDCM(sub['Lung'], sub['R'], sub['Theta'], sub['Phi'], nargout=0)
            pdcm.terminate()
            logger.info(f'Inverse POISSON modified, corrected Theta, and Phi images were generated.')

            affine, int_header, float_header = PLF.make_affine_and_headers(resolution, ni.load(sub['CT']).header)

            R = PLF.load_image(sub['R'])
            in_range, minV, maxV = PLF.check_values_minmax(R, expected_minV=0, expected_maxV=1)
            if not in_range:
                logger.warning(f'Poisson distance is expected to be between [0, 1], but it is in [{minV}, {maxV}]. The distance will be clipped to expected range.')
                R = np.clip(R, 0, 1)
            PLF.save_image(sub['R'], R, np.float32, float_header, affine)

            Theta = PLF.load_image(sub['Theta'])
            in_range, minV, maxV = PLF.check_values_minmax(Theta, expected_minV=-np.pi, expected_maxV=np.pi)
            if not in_range:
                logger.warning(f'Theta map is expected to be between [-pi, pi], but it is in [{minV}, {maxV}]. The Theta mape will be clipped to expected range.')
                Theta = np.clip(Theta, -np.pi, np.pi)
            PLF.save_image(sub['Theta'], Theta, np.float32, float_header, affine)

            Phi = PLF.load_image(sub['Phi'])
            in_range, minV, maxV = PLF.check_values_minmax(Phi, expected_minV=-np.pi/2, expected_maxV=np.pi/2)
            if not in_range:
                logger.warning(f'Phi map is expected to be between [-pi/2, pi/2], but it is in [{minV}, {maxV}]. The Phi mape will be clipped to expected range.')
                Phi = np.clip(Phi, -np.pi/2, np.pi/2)
            PLF.save_image(sub['Phi'], Phi, np.float32, float_header, affine)

            logger.info(f'Header of PDCM images were corrected and successfully saved.')

            SPIdx = PLF.convert_to_SPIdx(R, Theta, Phi)
            SPIdx = (SPIdx * (PLF.load_image(sub['Lung']) > 0))
            in_range, uniqueV = PLF.check_values_unique(SPIdx, expected_uniqueV=[i for i in range(37)])
            if not in_range:
                logger.warning(f'Spatial indexp map is expected to have all intiger values between [0, 36] , but it has {list(uniqueV)}. There are no remedy for it and process will continue, but check if something is wrong.')
            PLF.save_image(sub['SpatialIndex'], SPIdx, np.int16, int_header, affine)

            logger.info(f'Dense Spatial Index is generated.')
            
            PLF.plot([R, Theta, Phi, SPIdx], 
                     suptitle=f'{Cohort} - {Study} - {ID} - PDCM and Spatial Index',
                     save_path=f'{DIR}/QC_Figures/{ID}_ProcessedPDCM.png',
                     titles=['R', 'Theta', 'Phi', 'Spatial Index'],
                     cmaps=['jet', 'jet', 'jet', 'jet'],
                     minvs=[0, -np.pi, -np.pi/2, 0],
                     maxvs=[1, np.pi, np.pi/2, 36],
                     isDiscrete=[False, False, False, True],
                     colorbarTitles=['Distance', 'Radian', 'Radian', 'Spatial Index'],
                     show=False,
                    )

            logger.info(f'Screen shots are availabel in {DIR}/QC_Figures.')

            logger.info(f'Process is complete now.')
            PLF.close_logger(logger)
            return True
        except Exception as e:
            PLF.close_logger(logger)
            raise e
            
    # Make ROI-levl volumetric measures
    def Make_ROIlvl_Measures(sub, overwrite=False):
        spl_diam = 25 # mm
        air_threshold = -950 # HU.
        try:
            Cohort = sub['Cohort']
            Study = sub['Study']
            ID = sub['ID']
            DIR = sub['DIR']
            resolution = sub['Resolution']
            logger = PLF.get_logger(f'{DIR}/logs', f'{ID}_ROIVolumes')
            logger.info(f'Creating ROI-level volumetric measures started for subject {ID} of cohort {Cohort} - {Study}.')
            
            if not overwrite:
                should_continue = True
                for img_key in ['LungVolume', '950Volume', 'PercentEmph']:
                    if os.path.isfile(sub[img_key]):
                        logger.error(f'Output directory contains {img_key} image in {sub[img_key]}. Please remove then re-run ROI-level volumetric maker.')
                        should_continue = False
                if not should_continue:
                    PLF.close_logger(logger)
                    return False

            ROI_size = np.ceil(spl_diam/np.array(resolution)).astype('int')
            logger.info(f'Optimal ROI size for this subject is {tuple(ROI_size)}.')

            CT, header, _ = PLF.load_image(sub['CT'], returnHeaderInfo=True)
            Lung = ni.load(sub['Lung']).get_fdata().astype('int') > 0
            Airway = ni.load(sub['Airway']).get_fdata().astype('int') > 0
            Lung_ROI = np.ones_like(CT) * 10000
            Lung_ROI[Lung] = CT[Lung]
            Lung_ROI[Airway] = 10000
            del CT, Airway
            Lung_vol = uniform_filter(Lung.astype('float32'), size=ROI_size)*np.prod(ROI_size)*Lung
            Under_Threshold = uniform_filter((Lung_ROI < air_threshold).astype('float32'), size=ROI_size)*np.prod(ROI_size)*Lung
            Pct_emph = np.zeros_like(Lung_ROI)
            Pct_emph[Lung] = (100 * Under_Threshold[Lung] / Lung_vol[Lung]).astype('float32')

            logger.info(f'Dense ROI-level volumetric measures are calculated.')

            affine, int_header, float_header = PLF.make_affine_and_headers(resolution, CT_header=header)

            PLF.save_image(sub['LungVolume'], Lung_vol, 'int', header=int_header, affine=affine)
            PLF.save_image(sub['950Volume'], Under_Threshold, 'int', header=int_header, affine=affine)
            PLF.save_image(sub['PercentEmph'], Pct_emph, 'float32', header=float_header, affine=affine)


            logger.info(f'Dense ROI-level volumetric measures are stored in {DIR}.')

            PLF.plot([Lung_vol, Under_Threshold, Pct_emph], 
                     suptitle=f'{Cohort} - {Study} - {ID} - ROI-level volumetric measures',
                     save_path=f'{DIR}/QC_Figures/{ID}_ProcessedROILvlVol.png',
                     titles=['Lung volume', '-950 HU volume', '%emph-950'],
                     colorbarTitles=['Voxels', 'Voxels', '%'],
                     globalMinMax=False,
                     show=False,
                    )

            logger.info(f'Screen shots are availabel in {DIR}/QC_Figures.')

            logger.info(f'Process is complete now.')
            PLF.close_logger(logger)
            return True
        except Exception as e:
            PLF.close_logger(logger)
            raise e
            
    def Make_sLTP(sub, overwrite=False):
        spl_diam = 25 # mm
        try:
            from cupyx.scipy.ndimage import uniform_filter
            g_codebook = cp.asanyarray(loadmat('/home/soroush/Projects/Meta_data/upright_codebook.mat')['codebook'], dtype='float32')
            cents = loadmat('/home/soroush/Projects/Meta_data/cents.mat')
            cent_T = cp.asanyarray(cents['cent_T'])
            cent_S = cp.asanyarray(cents['cent_S'])

            Cohort = sub['Cohort']
            Study = sub['Study']
            ID = sub['ID']
            DIR = sub['DIR']
            resolution = sub['Resolution']
            logger = PLF.get_logger(f'{DIR}/logs', f'{ID}_sLTP')
            logger.info(f'Creating dense T3D and sLTP masks (original and corrected) started for subject {ID} of cohort {Cohort} - {Study}.')

            if not overwrite:
                should_continue = True
                for img_key in ['T3D', 'sLTP', 'sLTP36']:
                    if os.path.isfile(sub[img_key]):
                        logger.error(f'Output directory contains {img_key} image in {sub[img_key]}. Please remove then re-run sLTP maker.')
                        should_continue = False
                if not should_continue:
                    PLF.close_logger(logger)
                    return False

            ROI_size = np.ceil(spl_diam/np.array(resolution)).astype('int')
            logger.info(f'Optimal ROI size for this subject is {tuple(ROI_size)}.')

            CT, header, affine = PLF.load_image(sub['CT'], dtype=np.float32, returnHeaderInfo=True)
            Lung = PLF.load_image(sub['Lung'], dtype=np.int16)
            AW = PLF.load_image(sub['Airway'], dtype=np.int16)
            CT_Shape = CT.shape
            CT = cp.asanyarray(CT, dtype='float32')
            Lung = cp.asanyarray(Lung > 0)
            AW = cp.asanyarray(AW > 0)
            CT[AW] = 10000
            CT[cp.logical_not(Lung)] = 10000
            CT = cp.clip(CT, -1000, -400)
            CT = (CT + 1000)/600
            del AW

            affine, int_header, float_header = PLF.make_affine_and_headers(resolution, CT_header=header)

            T3D = np.ones(CT_Shape, dtype='int16') * 40
            patches_shape = np.array(CT_Shape) // 3
            step = 1000000
            for m in range(-1,2):
                for n in range(-1,2):
                    for p in range(-1,2):
                        CT = cp.roll(CT, shift=[m, n, p], axis=[0, 1, 2])
                        patches = view_as_windows(CT, window_shape=(3,3,3), step=3)
                        patches = cp.reshape(patches, (-1,1,27))
                        t3d_array = np.zeros((len(patches),))
                        i = -1
                        for i in range(patches.shape[0]//step):
                            t3d_array[i*step:(i+1)*step] = cp.argmin(cp.sqrt(cp.sum(cp.square(patches[i*step:(i+1)*step,:,:] - g_codebook), axis=-1)), axis=-1).get() + 1
                        if (i+1)*step < len(t3d_array):
                            t3d_array[(i+1)*step:] = cp.argmin(cp.sqrt(cp.sum(cp.square(patches[(i+1)*step:,:,:] - g_codebook), axis=-1)), axis=-1).get() + 1

                        T3D[1-m:patches_shape[0]*3:3, 1-n:patches_shape[1]*3:3, 1-p:patches_shape[2]*3:3] = t3d_array.reshape(patches_shape)
                        CT = cp.roll(CT, shift=[-m, -n, -p], axis=[0, 1, 2])
                        del patches

            del CT
            PLF.save_image(sub['T3D'], T3D, 'int', header=int_header, affine=affine)
            logger.info(f'Dense T3D calculated.')

            Lung = Lung.astype('int16')

            lam = 1.52
            w = 0.0048
            fact = lam*w
            SPIdx = PLF.load_image(sub['SpatialIndex'], dtype=np.int16)
            SPIdx = cp.asanyarray(SPIdx, dtype='int16')*Lung
            T3D = cp.asanyarray(T3D, dtype='int16')*Lung

            lung_vol = cp.copy(Lung).astype('float32')
            filt_size = [36, 36, 36]
            lung_vol = cp.clip(uniform_filter(lung_vol, size=filt_size), a_min=1e-5, a_max=None)
            dist = cp.zeros((*Lung.shape, 10), dtype='float32')
            for j in range(10):
                d = cp.zeros_like(Lung, dtype='float32')
                for i in range(40):
                    d += cp.square(uniform_filter((T3D == i+1).astype('float32'), size=filt_size)/lung_vol - cent_T[j,i])
                    if i < 36:
                        d += fact * cp.square((SPIdx == i+1).astype('float32') - cent_S[j,i])
                dist[:,:,:,j] = d
            sLTP = (((10 - cp.argmin(dist, axis=-1))*Lung).get()).astype('int16')
            del dist, lung_vol
            PLF.save_image(sub['sLTP36'], sLTP, 'int16', header=int_header, affine=affine)
            logger.info(f'Dense sLTP calculated.')


            lung_vol = cp.copy(Lung).astype('float32')
            filt_size = tuple(ROI_size)
            lung_vol = cp.clip(uniform_filter(lung_vol, size=filt_size), a_min=1e-5, a_max=None)
            dist = cp.zeros((*Lung.shape, 10), dtype='float32')
            for j in range(10):
                d = cp.zeros_like(Lung, dtype='float32')
                for i in range(40):
                    d += cp.square(uniform_filter((T3D == i+1).astype('float32'), size=filt_size)/lung_vol - cent_T[j,i])
                    if i < 36:
                        d += fact * cp.square((SPIdx == i+1).astype('float32') - cent_S[j,i])
                dist[:,:,:,j] = d
            sLTPCorrected = (((10 - cp.argmin(dist, axis=-1))*Lung).get()).astype('int16')
            del dist, lung_vol, Lung
            PLF.save_image(sub['sLTP'], sLTPCorrected, 'int16', header=int_header, affine=affine)
            logger.info(f'Dense sLTP Corrected calculated.')        

            logger.info(f'Dense T3D and sLTPs (both versions) are stored in {DIR}.')

            T3D = T3D.get()

            PLF.plot([T3D, sLTP, sLTPCorrected], 
                     suptitle=f'{Cohort} - {Study} - {ID} - T3D and sLTP',
                     save_path=f'{DIR}/QC_Figures/{ID}_ProcessedT3DsLTP.png',
                     titles=['Texton', 'sLTP (Original)', 'sLTP (Corrected)'],
                     colorbarTitles=['Texton index', 'sLTP label', 'sLTP label'],
                     isDiscrete=[True, True, True],
                     show=False,
                    )

            logger.info(f'Screen shots of T3D and sLTPs are availabel in {DIR}/QC_Figures.')

            logger.info(f'Process is complete now.')
            PLF.close_logger(logger)
            return True
        except Exception as e:
            PLF.close_logger(logger)
            raise e

    
    def Make_ULN_sLTP(sub, overwrite=False):
        try:
            Cohort = sub['Cohort']
            Study = sub['Study']
            ID = sub['ID']
            DIR = sub['DIR']
            resolution = sub['Resolution']
            logger = PLF.get_logger(f'{DIR}/logs', f'{ID}_sLTP')
            logger.info(f'Creating ULN-based maksed sLTP')
            
            if 'ULN' not in sub:
                logger.error('ULN key is not defined for the subject.')
                PLF.close_logger(logger)
                return False
            
            if sub['ULN'] is None:
                logger.error('ULN value is not set for the subject.')
                PLF.close_logger(logger)
                return False

            if not overwrite:
                should_continue = True
                for img_key in ['ULN_Mask', 'ULN_sLTP']:
                    if os.path.isfile(sub[img_key]):
                        logger.error(f'Output directory contains {img_key} image in {sub[img_key]}. Please remove then re-run ULN-based masked sLTP maker.')
                        should_continue = False
                if not should_continue:
                    PLF.close_logger(logger)
                    return False
                
            ULN = sub['ULN']
            sLTP, header, affine = PLF.load_image(sub['sLTP'], dtype=np.int16, returnHeaderInfo=True)
            PctEmpth = PLF.load_image(sub['PercentEmph'], dtype=np.float32)
            ULN_mask = (PctEmpth > ULN).astype('int16')
            PLF.save_image(sub['ULN_Mask'], ULN_mask, dtype=np.int16, header=header, affine=affine)
            logger.info(f'ULN mask is saved.')
            
            sLTP_masked = sLTP * ULN_mask
            PLF.save_image(sub['ULN_sLTP'], sLTP_masked, dtype=np.int16, header=header, affine=affine)
            logger.info(f'ULN-based masked sLTP is saved.')
            
            logger.info(f'Process is complete now.')
            PLF.close_logger(logger)
            return True
        except Exception as e:
            PLF.close_logger(logger)
            raise e
            
    
    # All CPU processes
    def CPUPipeLine(sub, overwrite=False):
        # CPU-based functions
        for function in [PLF.QC_inputs, PLF.Make_PDCM, PLF.Make_ROIlvl_Measures]:
            if not function(sub, overwrite):
                return False
        sub['LungRange'] = PLF.extract_lung_bounding_box(sub)
        PLF.write_sub_info(sub)
        return True
    
    # All GPU processes
    def GPUPipeLine(sub, device, overwrite=False):
        # GPU-based function
        for function in [PLF.Make_sLTP]:
            with device:
                if not function(sub, overwrite):
                    return False
        return True
    
    
    # Helpers for full pipeline
    def _process_subject_CPU(sub, overwrite=False):
        tt = time.time()
        try:
            Cohort = sub['Cohort']
            Study = sub['Study']
            ID = sub['ID']
            DIR = sub['DIR']

            PLF.make_sub_directories(sub)
            PLF.write_sub_info(sub)
            if not PLF.CPUPipeLine(sub, overwrite):
                raise Exception(f'Failed due to internal error. Please refer to logs ({DIR}/logs) for more details.')

            return f'{Cohort} - {Study} - {ID} - Passed In {time.time() - tt}'
        except Exception as e:
            return f'{Cohort} - {Study} - {ID} - Failed In {time.time() - tt} - {str(e)}'
    
    def _process_subject_CPU_ULNMaskedsLTP(sub, overwrite=False):
        tt = time.time()
        try:
            Cohort = sub['Cohort']
            Study = sub['Study']
            ID = sub['ID']
            DIR = sub['DIR']

            if not PLF.Make_ULN_sLTP(sub, overwrite=overwrite):
                raise Exception(f'Failed due to internal error. Please refer to logs ({DIR}/logs) for more details.')

            return f'{Cohort} - {Study} - {ID} - Passed In {time.time() - tt}'
        except Exception as e:
            return f'{Cohort} - {Study} - {ID} - Failed In {time.time() - tt} - {str(e)}'
        
    def _process_subject_CPU_screenshots(sub, overwrite=False):
        tt = time.time()
        try:
            Cohort = sub['Cohort']
            Study = sub['Study']
            ID = sub['ID']
            DIR = sub['DIR']

            PLF.plot_screenshots(sub, show=False, returnImages=False)

            return f'{Cohort} - {Study} - {ID} - Passed In {time.time() - tt}'
        except Exception as e:
            return f'{Cohort} - {Study} - {ID} - Failed In {time.time() - tt} - {str(e)}'
        
    def _filter_subjects_on_results(subjects, results):
        new_subjects = list()
        for i, sub in enumerate(subjects):
            if results[i]:
                new_subjects.append(sub)
        return new_subjects
    
    
    # Full parallelized pipeline    
    def FullPipeLine(subjects, device, overwrite=False, n_cores=10, print_GPU_errors=True):
        # Initial argument parsing!!!
        subjects = subjects if isinstance(subjects, list) else list([subjects])
        initial_len = len(subjects)
        n_cores = min(initial_len, n_cores)
        
        # Parallelize CPU process
        print(f'1. Parralleled process for {len(subjects)} subjects started on {n_cores} CPU cores ...')
        fn = partial(PLF._process_subject_CPU, overwrite=overwrite)
        results = ParallelMe(fn, subjects, name=f'ParallelPipeline({initial_len})', n_cores=n_cores, print_errors=False).run(returnResults=True)
        subjects = PLF._filter_subjects_on_results(subjects, results) # Remove unsuccessful CPU subjects
        
        # Run GPU process on CPU successful subjects
        if len(subjects) > 0:
            print(f'2. GPU process for {len(subjects)} subjects started on {str(device)} ...')
            results = list()
            message = list()
            for i, sub in enumerate(tqdm(subjects)):
                try:
                    tt = time.time()
                    Cohort = sub['Cohort']
                    Study = sub['Study']
                    ID = sub['ID']
                    DIR = sub['DIR']

                    res = PLF.GPUPipeLine(sub, device, overwrite)

                    if res:
                        message.append(f'{Cohort} - {Study} - {ID} - Passed In {time.time() - tt} - (GPU process)')
                        results.append(True)
                    else:
                        raise Exception(f'Failed due to internal error (GPU process). Please refer to logs ({DIR}/logs) for more details.')
                except Exception as e:
                    message.append(f'{Cohort} - {Study} - {ID} - Failed In {time.time() - tt} - {str(e)} - (GPU process)')
                    results.append(False)

            # Write and print GPU logs
            if not os.path.isdir('./multiprocess_logs'):
                os.mkdir('./multiprocess_logs')
            now = datetime.now()
            current_time = now.strftime("%Y%m%d_%H%M%S")
            with open(f'./multiprocess_logs/GPUPipeine_{current_time}', 'w') as f:
                f.write('\n'.join(message))
            if print_GPU_errors:
                for i, mess in enumerate(message):
                    if not results[i]:
                        print(mess)

            subjects = PLF._filter_subjects_on_results(subjects, results) # Remove unsuccessful GPU subjects
        else:
            print(f'No further action is possible.')
            return
        
        if len(subjects) > 0:
            n_cores = min(len(subjects), n_cores) # Reinitialize number of cores needed
            # Again parallelize making ULN-based masked sLTP
            print(f'3. Masked sLTP based on ULN generation for {len(subjects)} subjects started on {n_cores} CPU cores ...')
            fn = partial(PLF._process_subject_CPU_ULNMaskedsLTP, overwrite=overwrite)
            results = ParallelMe(fn, subjects, name=f'ULNMaskedsLTP({len(subjects)})', n_cores=n_cores, print_errors=False).run(returnResults=True)
            subjects = PLF._filter_subjects_on_results(subjects, results) # Remove unsuccessful CPU subjects
        else:
            print(f'No further action is possible.')
            return
            
        if len(subjects) > 0:
            n_cores = min(len(subjects), n_cores) # Reinitialize number of cores needed
            # Again parallelize taking screenshots process
            print(f'4. Screenshot process for {len(subjects)} subjects started on {n_cores} CPU cores ...')
            ParallelMe(PLF._process_subject_CPU_screenshots, subjects, name=f'Screenshot({len(subjects)})', n_cores=n_cores, print_errors=False).run()
        
            print(f'Full pipeline started for {initial_len} subjects and {len(subjects)} subjects are processed successfully.' + ('' if len(subjects) == initial_len else '\nPlease check the logs for unsuccessful subjects.'))
        else:
            print(f'No further action is possible.')
            return

###################################################################################### End of class
          

class ParallelMe:
    def __init__(self, fn, inputs, n_cores=10, name='', suffix='', log_dir='./multiprocess_logs', print_errors=True, error_keys=['error','failed']):
        
        self.fn = fn
        self.inputs = inputs
        self.n_cores = n_cores
        self.print_errors = print_errors
        self.error_keys = error_keys
        self.total = len(self.inputs)
        
        now = datetime.now()
        current_time = now.strftime("%Y%m%d_%H%M%S")
        self.full_name = (self.fn.__name__ if name == '' else name) + '_' + suffix + ('_' if suffix != '' else '') + current_time
        
        self.log_dir = log_dir
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

    def run(self, returnResults=False):
        with Pool(self.n_cores) as p:
            self.results = list(tqdm(p.imap(self.fn, self.inputs), total=self.total))
        
        Results = list()
        for res in self.results:
            failed = False
            for key in self.error_keys:
                if key in res.lower():
                    failed = True
            if failed:
                Results.append(False)
                if self.print_errors:
                    print(res)
            else:
                Results.append(True)

        with open(f'{self.log_dir}/{self.full_name}.txt', 'w') as f:
            f.write('\n'.join(self.results))
            
        if returnResults:
            return Results
            
    def get_template():
        print(
            "def process_subject(sub):\ntt = time.time()\ntry:\nCohort = sub['Cohort']\nStudy = sub['Study']\nID = sub['ID']\nDIR = sub['DIR']\n...\nif not PLF.func(sub):\nraise Exception(f'Failed due to internal error. Please refer to logs ({DIR}/logs) for more details.')\n...\nreturn f'{Cohort} - {Study} - {ID} - Passed In {time.time() - tt}'\nexcept Exception as e:\nreturn f'{Cohort} - {Study} - {ID} - Failed In {time.time() - tt} - {str(e)}'"
        )
        
    def clean_logs(log_dir='./multiprocess_logs'):
        if os.path.isdir(log_dir):
            sh.rmtree('./multiprocess_logs')
        
        
        
        
def view_as_windows(arr_in, window_shape, step=1):
    """Rolling window view of the input n-dimensional array.
    Windows are overlapping views of the input array, with adjacent windows
    shifted by a single row or column (or an index of a higher dimension).
    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    window_shape : integer or tuple of length arr_in.ndim
        Defines the shape of the elementary n-dimensional orthotope
        (better know as hyperrectangle [1]_) of the rolling window view.
        If an integer is given, the shape will be a hypercube of
        sidelength given by its value.
    step : integer or tuple of length arr_in.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.
    Returns
    -------
    arr_out : ndarray
        (rolling) window view of the input array.
    Notes
    -----
    One should be very careful with rolling views when it comes to
    memory usage.  Indeed, although a 'view' has the same memory
    footprint as its base array, the actual array that emerges when this
    'view' is used in a computation is generally a (much) larger array
    than the original, especially for 2-dimensional arrays and above.
    For example, let us consider a 3 dimensional array of size (100,
    100, 100) of ``float64``. This array takes about 8*100**3 Bytes for
    storage which is just 8 MB. If one decides to build a rolling view
    on this array with a window of (3, 3, 3) the hypothetical size of
    the rolling view (if one was to reshape the view for example) would
    be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
    even worse as the dimension of the input array becomes larger.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hyperrectangle
    Examples
    --------
    >>> import cupy as cp
    >>> from skimage.util.shape import view_as_windows
    >>> A = cp.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> window_shape = (2, 2)
    >>> B = view_as_windows(A, window_shape)
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[1, 2],
           [5, 6]])
    >>> A = cp.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> window_shape = (3,)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (8, 3)
    >>> B
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])
    >>> A = cp.arange(5*4).reshape(5, 4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19]])
    >>> window_shape = (4, 3)
    >>> B = view_as_windows(A, window_shape)
    >>> B.shape
    (2, 2, 4, 3)
    >>> B  # doctest: +NORMALIZE_WHITESPACE
    array([[[[ 0,  1,  2],
             [ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14]],
            [[ 1,  2,  3],
             [ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15]]],
           [[[ 4,  5,  6],
             [ 8,  9, 10],
             [12, 13, 14],
             [16, 17, 18]],
            [[ 5,  6,  7],
             [ 9, 10, 11],
             [13, 14, 15],
             [17, 18, 19]]]])
    """

    # -- basic checks on arguments
    if not isinstance(arr_in, cp.ndarray):
        raise TypeError("`arr_in` must be a cupy ndarray")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = arr_in.shape
    window_shape = tuple([int(w) for w in window_shape])

    if any(s < ws for s, ws in zip(arr_shape, window_shape)):
        raise ValueError("`window_shape` is too large")

    if any(ws < 0 for ws in window_shape):
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    win_indices_shape = tuple(
        [(s - ws) // st + 1 for s, ws, st in zip(arr_shape, window_shape, step)]
    )
    new_shape = win_indices_shape + window_shape

    window_strides = arr_in.strides
    indexing_strides = arr_in[slices].strides
    strides = indexing_strides + window_strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out