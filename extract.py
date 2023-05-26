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

'Refactor saving part of extract_tiff'

#%% Names ---------------------------------------------------------------------

# czi_name = 'Masina_CD7_(3c-540s)_20x_16bits.czi'
# czi_name = 'Stebler_CD7_(4c-120s)_5x-2x_14bits.czi'
# czi_name = 'Stoma_CD7_(3z-4c-240s)_5x-2x_16bits.czi'
# czi_name = 'Bertet_880_(100t-10z)_40x_8bits.czi'
# czi_name = 'Bertet_880_(566t-15z)_40x_8bits.czi'
# czi_name = 'Lelouard_780_(14z-6c)_40x_8bits.czi'
czi_name = 'Sidor_880_(6z-4c)_100x_16bits.czi'
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
    
    # Extract metadata
    with pyczi.open_czi(czi_path) as czidoc:    
        md_all = czidoc.metadata['ImageDocument']['Metadata']
        scn_coords = czidoc.scenes_bounding_rectangle
    md_img = (md_all['Information']['Image'])
    md_pix = (md_all['Scaling']['Items']['Distance'])
    md_chn = (md_img['Dimensions']['Channels']['Channel'])
    md_scn = (md_img['Dimensions']['S']['Scenes']['Scene'])
    
    # Read dimensions  
    nT = int(md_img['SizeT']) if 'SizeT' in md_img else 1
    nZ = int(md_img['SizeZ']) if 'SizeZ' in md_img else 1
    nC = int(md_img['SizeC']) if 'SizeC' in md_img else 1
    nY = int(md_img['SizeY']) if 'SizeY' in md_img else 1
    nX = int(md_img['SizeX']) if 'SizeX' in md_img else 1
    nS = int(md_img['SizeS']) if 'SizeS' in md_img else 1

    # Read general info
    bit_depth = md_img['ComponentBitCount']

    # Read pixel info
    pix_size, pix_dims = [], []
    for pix in md_pix:
        pix_size.append(pix['Value'])
        pix_dims.append(pix['@Id'])
    pix_size = tuple(pix_size)   
    pix_dims = tuple(pix_dims)  

    # Read channel info
    chn_name = []
    for chn in range(nC):
        if nC <= 1: chn_name.append(md_chn['@Name'])
        else: chn_name.append(md_chn[chn]['@Name'])
    chn_name = tuple(chn_name) 

    # Read scene info
    scn_name, snY, snX, sY0, sX0 = [], [], [], [], [],
    if nS > 1:   
        for scn in range(nS):
            tmp_name = md_scn[scn]['ArrayName'] + '-' + md_scn[scn]['@Name']
            tmp_name = re.sub(r'\d+', lambda m: m.group().zfill(2), tmp_name)
            scn_name.append(tmp_name)        
            snY.append(scn_coords[scn][3]) 
            snX.append(scn_coords[scn][2]) 
            sY0.append(scn_coords[scn][1]) 
            sX0.append(scn_coords[scn][0]) 

    # Append metadata dict      
    metadata = {    
        'nT': nT, 'nZ': nZ, 'nC': nC, 'nY': nY, 'nX': nX, 'nS': nS, 
        'bit_depth': bit_depth,
        'pix_size': pix_size, 'pix_dims': pix_dims,
        'chn_name': chn_name,
        'scn_name': scn_name, 'snY': snY, 'snX': snX, 'sY0': sY0, 'sX0': sX0,
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

#%% Function: extract_tiff ----------------------------------------------------

# Arguments
czi_path = czi_path
rT = 'all'
rZ = 'all'
rC = 'all'
zoom = 0.1
hyperstack = True

start = time.time()
print('Extract data')

# Extract data
metadata, data = extract_data(czi_path, rT=rT, rZ=rZ, rC=rC, zoom=zoom)

end = time.time()
print(f'  {(end-start):5.3f} s') 

# -----------------------------------------------------------------------------

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

# 
if hyperstack:
    
    for scn, data in enumerate(data): 
        if scn == 0: 
            scn_name = '' 
        else: scn_name = '_' + metadata['scn_name'][scn]            
        
        data_path = Path(dir_path, 
            Path(czi_path).stem + f'{scn_name}.tif'
            )
        
        io.imsave(
            data_path,
            data.astype(f"uint{metadata['bit_depth']}"),
            check_contrast=False, imagej=True,
            metadata={'axes': 'TZCYX'}
            )
    
#     if isinstance(data, np.ndarray):
#         for scn, data in enumerate(data): 
#             if scn == 0:
#                 scn_name = ''
#             else:
#                 scn_name = '_' + metadata['scn_name'][scn]            
            
#             data_path = Path(dir_path, 
#                 Path(czi_path).stem + f'{scn_name}.tif'
#                 )
            
#             io.imsave(
#                 data_path,
#                 data.astype(f"uint{metadata['bit_depth']}"),
#                 check_contrast=False, imagej=True,
#                 metadata={'axes': 'TZCYX'}
#                 )
    
#     if isinstance(data, list):     
#         for scn, data in enumerate(data): 
#             scn_name = metadata['scn_name'][scn]
#             data_path = Path(dir_path, 
#                 Path(czi_path).stem + f'{scn_name}.tif'
#                 )
#             io.imsave(
#                 data_path,
#                 data.astype(f"uint{metadata['bit_depth']}"),
#                 check_contrast=False, imagej=True,
#                 metadata={'axes': 'TZCYX'}
#                 )

# else:
    
    
    
#     if isinstance(data, np.ndarray):
#         for t in range(data.shape[0]):
#             for z in range(data.shape[1]):
#                 for c in range(data.shape[2]):
#                     data_path = Path(dir_path, 
#                         Path(czi_path).stem + f'_t{t}-z{z}-c{c}.tif'
#                         )
#                     io.imsave(
#                         data_path,
#                         data[t, z, c, ...].astype(f"uint{metadata['bit_depth']}"),
#                         check_contrast=False, imagej=True,
#                         metadata={'axes': 'YX'}
#                         )    
                    
#     if isinstance(data, list):
#         for scn, data in enumerate(data):
#             scn_name = metadata['scn_name'][scn]
#             for t in range(data.shape[0]):
#                 for z in range(data.shape[1]):
#                     for c in range(data.shape[2]):
#                             data_path = Path(dir_path, 
#                                 Path(czi_path).stem + f'_{scn_name}_t{t}-z{z}-c{c}.tif'
#                                 )
#                             io.imsave(
#                                 data_path,
#                                 data[t, z, c, ...].astype(f"uint{metadata['bit_depth']}"),
#                                 check_contrast=False, imagej=True,
#                                 metadata={'axes': 'YX'}
#                                 )
                    


#%% Execute -------------------------------------------------------------------
   
# start = time.time()
# print('Extract data')

# metadata, data = extract_data(czi_path, rT='all', rZ='all', rC='all', zoom=0.1)

# end = time.time()
# print(f'  {(end-start):5.3f} s') 

#%% Save ----------------------------------------------------------------------

# io.imsave(
#     czi_path.replace('.czi', '.tif'),
#     data.astype('uint16'), check_contrast=False, imagej=True,
#     metadata={'axes': 'TZCYX'}
#     )

# digits = len(str(len(data)))
