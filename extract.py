#%% Imports -------------------------------------------------------------------

import re
import sys
import time
import numpy as np
from skimage import io 
from pathlib import Path
from itertools import product
from pylibCZIrw import czi as pyczi
from joblib import Parallel, delayed 

#%% Comments ------------------------------------------------------------------

'Add padding solution to save_preview'
'Tile images for save_preview'
'Manage datatype outputs'

#%% Names ---------------------------------------------------------------------

czi_name = 'Masina_CD7_(3c-540s)_20x_16bits.czi'
# czi_name = 'Stebler_CD7_(4c-120s)_5x-2x_14bits.czi'
# czi_name = 'Stoma_CD7_(3z-4c-240s)_5x-2x_16bits.czi'
# czi_name = 'Bertet_880_(100t-10z)_40x_8bits.czi'
# czi_name = 'Bertet_880_(566t-15z)_40x_8bits.czi'
# czi_name = 'Lelouard_780_(14z-6c)_40x_8bits.czi'
# czi_name = 'Sidor_880_(6z-4c)_100x_16bits.czi'
# czi_name = 'Bruneau_Z2_(11z-3c)_20x_8bits_Stitching.czi'
# czi_name = 'Aggad_880_(5z-3c)_40x_8bits.czi'

#%% Initialize ----------------------------------------------------------------

czi_path = str(Path('data') / czi_name)

#%% Function: extract_metadata ------------------------------------------------

def extract_metadata(czi_path):
    
    """ 
    Extract and reformat metadata from czi file.
    
    Parameters
    ----------
    czi_path : str 
        Path to the czi file.
    
    Returns
    -------  
    metadata : dict
        Reformated metadata.
        
    """
    
    # Find key in nested dictionary (first occurence)
    def find_key(dictionary, target_key):
        for key, value in dictionary.items():
            if key == target_key:
                return value
            elif isinstance(value, dict):
                result = find_key(value, target_key)
                if result is not None:
                    return result
    
    # Extract metadata
    with pyczi.open_czi(czi_path) as czidoc:    
        md_all = czidoc.metadata['ImageDocument']['Metadata']
        scn_coords = czidoc.scenes_bounding_rectangle
    md_img = (md_all['Information']['Image'])
    md_pix = (md_all['Scaling']['Items']['Distance'])
    md_chn = (md_img['Dimensions']['Channels']['Channel'])
    md_scn = (md_img['Dimensions']['S']['Scenes']['Scene'])
    md_time = find_key(md_all, 'TimeSpan')
    
    # Read dimensions  
    nT = int(md_img['SizeT']) if 'SizeT' in md_img else 1
    nZ = int(md_img['SizeZ']) if 'SizeZ' in md_img else 1
    nC = int(md_img['SizeC']) if 'SizeC' in md_img else 1
    nY = int(md_img['SizeY']) if 'SizeY' in md_img else 1
    nX = int(md_img['SizeX']) if 'SizeX' in md_img else 1
    nS = int(md_img['SizeS']) if 'SizeS' in md_img else 1
    
    # Read general info
    bit_depth = int(md_img['PixelType'][4:])
    
    # Read pixel info
    pix_size, pix_dims = [], []
    if len(md_pix) == 2:   
        pix_size = tuple((
            float(md_pix[0]['Value']), 
            float(md_pix[1]['Value']), 
            1
            ))   
    if len(md_pix) == 3: 
        pix_size = tuple((
            float(md_pix[0]['Value']), 
            float(md_pix[1]['Value']), 
            float(md_pix[2]['Value'])
            )) 
    pix_dims = tuple(('x', 'y', 'z'))  
    
    # Read time info
    if md_time is not None:
        time_interval = float(md_time['Value'])
        if md_time['DefaultUnitFormat']  == 'ms':
            time_interval /= 1000
        if md_time['DefaultUnitFormat']  == 'min':
            time_interval *= 60
        elif md_time['DefaultUnitFormat']  == 'h':
            time_interval *= 3600
    else:
        time_interval = None
    
    # Read channel info
    chn_name = []
    for chn in range(nC):
        if nC <= 1: chn_name.append(md_chn['@Name'])
        else: chn_name.append(md_chn[chn]['@Name'])
    chn_name = tuple(chn_name) 
    
    # Read scene info
    scn_well, scn_pos, snY, snX, sY0, sX0 = [], [], [], [], [], []
    if nS > 1:   
        for scn in range(nS):
            tmp_well = md_scn[scn]['ArrayName']
            tmp_well = f'{tmp_well[0]}{int(tmp_well[1:]):02}'
            tmp_pos = md_scn[scn]['@Name']
            tmp_pos = int(tmp_pos[1:])
            scn_well.append(tmp_well)
            scn_pos.append(tmp_pos)
            snY.append(scn_coords[scn][3]) 
            snX.append(scn_coords[scn][2]) 
            sY0.append(scn_coords[scn][1]) 
            sX0.append(scn_coords[scn][0]) 
    
    # Append metadata dict      
    metadata = {    
        'nT': nT, 'nZ': nZ, 'nC': nC, 'nY': nY, 'nX': nX, 'nS': nS, 
        'bit_depth': bit_depth,
        'pix_size': pix_size, 'pix_dims': pix_dims,
        'time_interval': time_interval,
        'chn_name': chn_name,
        'scn_well': scn_well, 'scn_pos': scn_pos, 
        'snY': snY, 'snX': snX, 'sY0': sY0, 'sX0': sX0,
        }
    
    return metadata

#%% Function: extract_data ----------------------------------------------------

def extract_data(czi_path, rT='all', rZ='all', rC='all', zoom=1):

    """ 
    Extract data (images) from czi file.
    
    Parameters
    ----------
    czi_path : str 
        Path to the czi file.
        
    rT : str, int or tuple of int
        Requested timepoint(s).
        To select all timepoint(s) use 'all'.
        To select some timepoint(s) use tuple of int : expl (0, 1, 4).
        To select a specific timepoint use int : expl 0.
        
    rZ : str, int or tuple of int
        Requested timepoint(s).
        Selection rules see rT.
        
    rC : str, int or tuple of int
        Requested timepoint(s).
        Selection rules see rT.
            
    zoom : float
        Downscaling factor for extracted images.
        From 0 to 1, 1 meaning no downscaling.
            
    Returns
    -------  
    metadata : dict
        Reformated metadata.
        
    data : ndarray or list of ndarray
        Images extracted as hyperstack(s).
        
    """

    # Extract metadata
    metadata = extract_metadata(czi_path)
    
    # Format request
    def format_request(dim, nDim, name):
        if dim == 'all':
            dim = np.arange(nDim)
        elif isinstance(dim, tuple):
            dim = np.array(dim)
        elif isinstance(dim, int):
            dim = np.array([dim]) 
        if np.any(dim > nDim - 1):
            print(f'Wrong {name} request')
            sys.exit()
        return dim 
    
    rT = format_request(rT, metadata['nT'], 'timepoint(s)')
    rZ = format_request(rZ, metadata['nZ'], 'slice(s)')
    rC = format_request(rC, metadata['nC'], 'channel(s)')
    
    # Determine extraction pattern
    tzc_pat = list(product(rT, rZ, rC))
    tzc_pat = np.array(tzc_pat)
    tzc_idx = np.empty_like(tzc_pat)
    for i in range(tzc_pat.shape[1]):
        _, inverse = np.unique(tzc_pat[:, i], return_inverse=True)
        tzc_idx[:, i] = inverse 
        
    def _extract_data(scn):
     
        if metadata['nS'] <= 1:
            x0 = 0; snX = metadata['nX']
            y0 = 0; snY = metadata['nY'] 
        else:
            x0 = metadata['sX0'][scn]; snX = metadata['snX'][scn]
            y0 = metadata['sY0'][scn]; snY = metadata['snY'][scn]
            
        # Preallocate data
        data = np.zeros((
            rT.size, rZ.size, rC.size,
            int(snY * zoom), int(snX * zoom)
            ), dtype=int)
        
        # Extract data
        with pyczi.open_czi(czi_path) as czidoc:     
            for pat, idx in zip(tzc_pat, tzc_idx):
                data[idx[0], idx[1], idx[2], ...] = czidoc.read(
                    roi=(x0, y0, snX, snY), 
                    plane={'T': pat[0], 'Z': pat[1], 'C': pat[2]}, 
                    zoom=zoom,
                    ).squeeze()
                
        return data
    
    # Run extract_data
    if metadata['nS'] > 1:
        outputs = Parallel(n_jobs=-1)(
            delayed(_extract_data)(scn) 
            for scn in range(metadata['nS'])
            )
    else:
        outputs = [_extract_data(scn)
            for scn in range(metadata['nS'])
            ]
        
    # Extract outputs
    data = [data for data in outputs]
    if len(data) == 1: data = data[0]
    
    return metadata, data

#%% Function: save_tiff -------------------------------------------------------

def save_tiff(czi_path, rT='all', rZ='all', rC='all', zoom=1, hyperstack=True):

    """ 
    Extract and save scenes from a czi file as 
    ImageJ compatible tiff hyperstack or images.    
    Saved files are stored new folder named like 
    the czi file without extension.
    
    Parameters
    ----------
    czi_path : str 
        Path to the czi file.
        
    rT : str, int or tuple of int
        Requested timepoint(s).
        To select all timepoint(s) use 'all'.
        To select some timepoint(s) use tuple of int : expl (0, 1, 4).
        To select a specific timepoint use int : expl 0.
        
    rZ : str, int or tuple of int
        Requested timepoint(s).
        Selection rules see rT.
        
    rC : str, int or tuple of int
        Requested timepoint(s).
        Selection rules see rT.
            
    zoom : float
        Downscaling factor for extracted images.
        From 0 to 1, 1 meaning no downscaling.
        
    hyperstack : bool
        If True, images are saved as hyperstacks.
        If False, images are saved individually.
                    
    """    

    # Extract data
    metadata, data = extract_data(czi_path, rT=rT, rZ=rZ, rC=rC, zoom=zoom)
    
    # Setup saving directory
    def setup_directory(czi_path):
        czi_name = Path(czi_path).name
        dir_name = Path(czi_path).stem
        dir_path = Path(czi_path.replace(czi_name, dir_name))
        if dir_path.is_dir():
            for item in dir_path.iterdir():
                if item.is_dir():
                    setup_directory(item)
                    item.rmdir()
                else:
                    item.unlink()    
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    dir_path = setup_directory(czi_path)

    # Save scenes as hyperstacks or separated images 
    for scn in range(len(data)):
        
        pix_size_x = metadata['pix_size'][0] / zoom * 1e06 
        pix_size_y = metadata['pix_size'][1] / zoom * 1e06
        pix_size_z = metadata['pix_size'][2] * 1e06
        time_interval = metadata['time_interval']
        
        if metadata['nS'] == 1: 
            scene = data
            scn_name = '' 
        else: 
            scene = data[scn]
            scn_well = metadata['scn_well'][scn]
            scn_pos = metadata['scn_pos'][scn]
            scn_name = f"_{scn_well}-{scn_pos:02}"

        if hyperstack:    
            
            scene_path = Path(dir_path, 
                Path(czi_path).stem + f'{scn_name}.tif'
                )
            
            io.imsave(
                scene_path,
                scene.astype(f"uint{metadata['bit_depth']}"),
                check_contrast=False, imagej=True,
                resolution=(1/pix_size_x, 1/pix_size_y),
                metadata={
                    'unit': 'um',
                    'spacing': pix_size_z,
                    'finterval': time_interval,
                    'axes': 'TZCYX'
                    }
                )
            
        else:
                       
            for t in range(scene.shape[0]):
                for z in range(scene.shape[1]):
                    for c in range(scene.shape[2]):
                        
                        scene_path = Path(dir_path, 
                            Path(czi_path).stem + f'{scn_name}_t{t}-z{z}-c{c}.tif'
                            )
                        
                        io.imsave(
                            scene_path,
                            scene[t, z, c, ...].astype(f"uint{metadata['bit_depth']}"),
                            check_contrast=False, imagej=True,
                            resolution=(1/pix_size_x, 1/pix_size_y),
                            metadata={
                                'unit': 'um',
                                'spacing': pix_size_z,
                                'finterval': time_interval,
                                'axes': 'YX'
                                }
                            ) 

#%% Function: save_preview ----------------------------------------------------

# Arguments
zoom = 0.05
noGap = True

# -----------------------------------------------------------------------------

# Extract data
metadata, data = extract_data(czi_path, rT='all', rZ='all', rC='all', zoom=zoom)

#%%

# 
if metadata['nS'] <= 1:
    print('Cannot preview single scene czi')
    sys.exit()
    
#
nT = metadata['nT']
nZ = metadata['nZ']
nC = metadata['nC']
   
#
well_data = []
for well in sorted(set(metadata['scn_well'])):
    
    #
    idx = [i for i, w in enumerate(metadata['scn_well']) if well in w]
    
    #
    sX0 = [int(metadata['sX0'][i] * zoom) for i in idx]
    sY0 = [int(metadata['sY0'][i] * zoom) for i in idx]
    snX = [int(metadata['snX'][i] * zoom) for i in idx]
    snY = [int(metadata['snY'][i] * zoom) for i in idx]    
    wX0 = [x0 - np.min(sX0) for x0 in sX0]
    wY0 = [y0 - np.min(sY0) for y0 in sY0]    
    wX1 = [x0 + nX for x0, nX in zip(wX0, snX)]
    wY1 = [y0 + nY for y0, nY in zip(wY0, snY)]    
       
    #
    tmp_data = np.zeros((nT, nZ, nC, np.max(wY1), np.max(wX1)), dtype=int)
    for i in range(len(idx)):
        tmp_data[..., wY0[i]:wY1[i], wX0[i]:wX1[i]] = data[idx[i]]        
    
    #
    if noGap:
        nonzero_rows = np.where(np.any(tmp_data[0,0,0,...], axis=1))[0]
        nonzero_cols = np.where(np.any(tmp_data[0,0,0,...], axis=0))[0]    
        tmp_data = tmp_data[..., nonzero_rows, :][..., nonzero_cols]   
    
    #
    well_data.append(tmp_data)
        
    
#%% Execute -------------------------------------------------------------------

# start = time.time()
# print('Extract metadata')

# metadata = extract_metadata(czi_path)

# end = time.time()
# print(f'  {(end-start):5.3f} s') 

# # -----------------------------------------------------------------------------
   
# start = time.time()
# print('Extract data')

# metadata, data = extract_data(czi_path, rT='all', rZ='all', rC='all', zoom=0.1)

# end = time.time()
# print(f'  {(end-start):5.3f} s') 

# # -----------------------------------------------------------------------------

# start = time.time()
# print('save tiff')

# # Extract data
# save_tiff(czi_path, rT='all', rZ='all', rC='all', zoom=0.1, hyperstack=True)

# end = time.time()
# print(f'  {(end-start):5.3f} s') 

#%% Save ----------------------------------------------------------------------

# io.imsave(
#     czi_path.replace('.czi', '.tif'),
#     data.astype('uint16'), check_contrast=False, imagej=True,
#     metadata={'axes': 'TZCYX'}
#     )

# digits = len(str(len(data)))
