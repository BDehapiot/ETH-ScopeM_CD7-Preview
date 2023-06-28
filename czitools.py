#%% Imports -------------------------------------------------------------------

import sys
import time
import numpy as np
from skimage import io 
from pathlib import Path
from itertools import product
from pylibCZIrw import czi as pyczi
from joblib import Parallel, delayed 
from PIL import Image, ImageDraw, ImageFont

#%% Comments ------------------------------------------------------------------

#%% Name czi ------------------------------------------------------------------

data_path = 'D:\local_CZITools\data'

# czi_name = 'Masina_CD7_(3c-540s)_20x_16bits.czi'
# czi_name = 'Stebler_CD7_(4c-120s)_5x-2x_14bits.czi'
# czi_name = 'Stoma_CD7_(3z-4c-240s)_5x-2x_16bits.czi'
# czi_name = 'Bertet_880_(100t-10z)_40x_8bits.czi'
# czi_name = 'Bertet_880_(566t-15z)_40x_8bits.czi'
czi_name = 'Lelouard_780_(14z-6c)_40x_8bits.czi'
# czi_name = 'Sidor_880_(6z-4c)_100x_16bits.czi'
# czi_name = 'Bruneau_Z2_(11z-3c)_20x_8bits_Stitching.czi'
# czi_name = 'Aggad_880_(5z-3c)_40x_8bits.czi'

#%% Initialize ----------------------------------------------------------------

czi_path = str(Path(data_path) / czi_name)

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
    md_time = find_key(md_all, 'TimeSpan')
    try:
        md_scn = (md_img['Dimensions']['S']['Scenes']['Scene'])
    except:       
        md_scn = None

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
    if nT > 1:
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
    scn_well, scn_pos = [], []
    snY, snX, sY0, sX0 = [], [], [], []
    sY0stage, sX0stage = [], []
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
            stage = md_scn[scn]['CenterPosition']
            sYstage = float(stage[stage.index(',')+1:-1])
            sXstage = float(stage[0:stage.index(',')])
            sY0stage.append(np.round(
                sYstage - (scn_coords[scn][3] * 0.5 * pix_size[0] * 1e06), 3))
            sX0stage.append(np.round(
                sXstage - (scn_coords[scn][2] * 0.5 * pix_size[0] * 1e06), 3))

    # Append metadata dict      
    metadata = {    
        'nT': nT, 'nZ': nZ, 'nC': nC, 'nY': nY, 'nX': nX, 'nS': nS, 
        'bit_depth': bit_depth,
        'pix_size': pix_size, 'pix_dims': pix_dims,
        'time_interval': time_interval,
        'chn_name': chn_name,
        'scn_well': scn_well, 'scn_pos': scn_pos, 
        'snY': snY, 'snX': snX, 'sY0': sY0, 'sX0': sX0,
        'sY0stage': sY0stage, 'sX0stage': sX0stage,
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
    
    # Run _extract_data
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
    Save scenes from a czi file as 
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

#%% Function: CD7_preview -----------------------------------------------------

def CD7_preview(
        czi_path, 
        rT='all', 
        rZ='all', 
        rC='all',
        zoom=1,
        uint8=True,
        labels=True, 
        label_size=1,
        no_gap=True,
        pad_size=1
        ):
    
    """ 
    Save a preview image of a multiwell CD7 acquisition as a tiff file 
    named like the czi + 'preview' suffix.
    
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
        
    uint8 : bool
        If True, and original data is uint16, convert to uint8

    labels : bool
        If True, add well and position labels 
    
    label_size : float
        Adjust label size (< 1 reduce size, >1 increase size)
        
    no_gap : bool
        If True, remove gaps between positions
        
    pad_size : int
        Pad well image with white pixels 
                    
    """
    
    # Extract data
    metadata, data = extract_data(czi_path, rT=rT, rZ=rZ, rC=rC, zoom=zoom)

    # Check if multiple scenes czi
    if metadata['nS'] <= 1:
        print('Cannot preview single scene czi')
        sys.exit()

    well_data = []
    well_names = sorted(set(metadata['scn_well']))
    for well_name in well_names:
        
        # Indexes
        idx = [i for i, w in enumerate(metadata['scn_well']) if well_name in w]
        
        # Spatial info
        sX0 = [int(metadata['sX0'][i] * zoom) for i in idx]
        sY0 = [int(metadata['sY0'][i] * zoom) for i in idx]
        snX = [int(metadata['snX'][i] * zoom) for i in idx]
        snY = [int(metadata['snY'][i] * zoom) for i in idx]    
        wX0 = [x0 - np.min(sX0) for x0 in sX0]
        wY0 = [y0 - np.min(sY0) for y0 in sY0]    
        wX1 = [x0 + nX for x0, nX in zip(wX0, snX)]
        wY1 = [y0 + nY for y0, nY in zip(wY0, snY)]    
           
        # Extract and process scenes
        tmp_data = np.zeros((
            metadata['nT'], metadata['nZ'], metadata['nC'], 
            np.max(wY1), np.max(wX1)
            ), dtype=int)
        
        for i in range(len(idx)):
            
            scene = data[idx[i]]
            
            # Convert to uint8
            if uint8 and metadata['bit_depth'] == 16:
                scene = scene // 255
                
            # Add labels
            if labels:
                
                width = scene.shape[-1]
                height = scene.shape[-2]  
                scn_well = metadata['scn_well'][idx[i]]
                scn_pos = metadata['scn_pos'][idx[i]]
                fill = 255 if uint8 else 65535
                
                font_well = ImageFont.truetype(
                    'arial.ttf', size=int(width // 4 * label_size)) 
                font_pos = ImageFont.truetype(
                    'arial.ttf', size=int(width // 6 * label_size)) 
                text_img = Image.new('I', (width, height), 'black')                
                draw = ImageDraw.Draw(text_img)
                
                if scn_pos == 1:                
                    draw.text(
                      (10, 5), f'{scn_well}', 
                      font=font_well, fill=fill,
                      ) 
                    
                draw.text(
                    (width - 10, height - 10), f'{scn_pos}', 
                    font=font_pos, fill=fill,
                    anchor='rb'
                    )
                
                np.maximum(scene, text_img, out=scene)
                
            tmp_data[..., wY0[i]:wY1[i], wX0[i]:wX1[i]] = scene

        # Remove gaps
        if no_gap:        
            nonzero_rows = np.where(np.any(tmp_data[0,0,0,...], axis=1))[0]
            nonzero_cols = np.where(np.any(tmp_data[0,0,0,...], axis=0))[0]    
            tmp_data = tmp_data[..., nonzero_rows[:, None], nonzero_cols]

        # Append well_data
        well_data.append(tmp_data)

    # Process well_data
    wnX = [wdata.shape[-1] for wdata in well_data]
    wnY = [wdata.shape[-2] for wdata in well_data]
    wnX_mode = np.argmax(np.bincount(np.array(wnX)))
    wnY_mode = np.argmax(np.bincount(np.array(wnY)))

    for i, wdata in enumerate(well_data):

        # Check for x shape consitency
        if wdata.shape[-1] > wnX_mode:
            wdata = wdata[..., 0:wnX_mode]
        if wdata.shape[-1] < wnX_mode:
            padX = wnX_mode - wdata.shape[-1]
            wdata = np.pad(wdata, (
                (0, 0), (0, 0), (0, 0),
                (0, 0), (padX, 0),
                ), constant_values=0)
        
        # Check for y shape consitency
        if wdata.shape[-2] > wnY_mode:
            wdata = wdata[..., 0:wnY_mode,:]
        if wdata.shape[-2] < wnY_mode:
            padY = wnY_mode - wdata.shape[-2]
            wdata = np.pad(wdata, (
                (0, 0), (0, 0), (0, 0),
                (padY, 0), (0, 0),
                ), constant_values=0)
                
        # Pad well border
        constant_values = 255 if uint8 else 65535
        wdata[..., :pad_size, :] = constant_values  # Left border
        wdata[..., -pad_size:, :] = constant_values  # Right border
        wdata[..., :, :pad_size] = constant_values  # Top border
        wdata[..., :, -pad_size:] = constant_values  # Bottom border
            
        well_data[i] = wdata
        
    # Count well rows and columns
    def count_rows_cols(strings_list):
        letters_set = set()
        digits_set = set()        
        for string in strings_list:
            letters = [char for char in string if char.isalpha()]
            digits = [char for char in string if char.isdigit()]
            letters_set.update(letters)
            digits_set.update(digits)
        nRow = len(letters_set)
        nCol = len(digits_set)
        return nRow, nCol
    nRow, nCol = count_rows_cols(well_names)  

    # Assemble preview
    preview = []
    well_data = [np.transpose(wdata, (3,4,0,1,2)) for wdata in well_data]
    for row in range(nRow):
        preview.append(np.hstack(well_data[nCol*row:nCol*row+nCol]))
    preview = np.vstack(preview)
    preview = np.transpose(preview, (2,3,4,0,1))
    well_data = [np.transpose(wdata, (2,3,4,0,1)) for wdata in well_data]

    # Save preview
    pix_size_x = metadata['pix_size'][0] / zoom * 1e06 
    pix_size_y = metadata['pix_size'][1] / zoom * 1e06
    pix_size_z = metadata['pix_size'][2] * 1e06
    time_interval = metadata['time_interval']
    dtype = 'uint8' if uint8 else 'uint16'
    io.imsave(
        czi_path.replace('.czi', '_preview.tif'),
        preview.astype(dtype),
        check_contrast=False, imagej=True,
        resolution=(1/pix_size_x, 1/pix_size_y),
        metadata={
            'unit': 'um',
            'spacing': pix_size_z,
            'finterval': time_interval,
            'axes': 'TZCYX'
            }
        )

#%% Execute -------------------------------------------------------------------

# start = time.time()
# print('Extract metadata')

# metadata = extract_metadata(czi_path)

# end = time.time()
# print(f'  {(end-start):5.3f} s') 

# # # -----------------------------------------------------------------------------
   
# start = time.time()
# print('Extract data')

# metadata, data = extract_data(czi_path, rT='all', rZ='all', rC='all', zoom=0.25)

# end = time.time()
# print(f'  {(end-start):5.3f} s') 

# # # -----------------------------------------------------------------------------

# start = time.time()
# print('Save tiff')

# save_tiff(czi_path, rT='all', rZ='all', rC='all', zoom=0.25, hyperstack=True)

# end = time.time()
# print(f'  {(end-start):5.3f} s') 

# # # -----------------------------------------------------------------------------

# start = time.time()
# print('CD7 preview')

# CD7_preview(czi_path, rT='all', rZ='all', rC='all', zoom=0.25,
#     uint8=True, labels=True, label_size=0.75, no_gap=True, pad_size=4
#     )

# end = time.time()
# print(f'  {(end-start):5.3f} s') 
