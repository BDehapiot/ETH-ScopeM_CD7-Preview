#%% Imports -------------------------------------------------------------------

import re
import time
import numpy as np
from skimage import io 
from pathlib import Path
from pylibCZIrw import czi as pyczi

#%% Parameters ----------------------------------------------------------------

c = 0
z = 0 
t = 0

zoom = 0.1
pad = 20 
noGap = True

# img_name = 'Digoxin_Bulfan_blm_mel5-01.czi'
img_name = 'test02-withsoftwareAF-20x1x-095-60wells-01.czi'
# img_name = 'HU4NZLpR-202304051711.czi'
img_path = str(Path('data') / img_name)

#%% Process -------------------------------------------------------------------

start = time.time()
print('Process')

with pyczi.open_czi(img_path) as czidoc:

    # Read metadata
    md_xml = czidoc.raw_metadata
    md_all = (czidoc.metadata['ImageDocument']['Metadata'])
    md_img = (md_all['Information']['Image'])
    md_scn = (md_img['Dimensions']['S']['Scenes']['Scene'])
    
    # Extract dimensions
    nX = int(md_img['SizeX'])
    nY = int(md_img['SizeY'])
    nZ = int(md_img['SizeZ'])
    nC = int(md_img['SizeC'])
    nS = int(md_img['SizeS'])
    
    # Extract scene data (sData)
    
    sData = {
        'sImg': [],
        'x0': [],
        'y0': [],
        'sWidth': [],
        'sHeight': [],
        'wID': [],
        'wTile': [],
        }
    
    sCoords = czidoc.scenes_bounding_rectangle    
    for s in range(len(sCoords)):
        
        # Scene info (x0 & y0 = top left pixel)
        x0 = sCoords[s][0]
        y0 = sCoords[s][1]
        sWidth = sCoords[s][2]
        sHeight = sCoords[s][3]
        wID = md_scn[s]['ArrayName']
        wTile = md_scn[s]['@Name']
        
        # Scene image
        sImg = czidoc.read(
            roi=(x0, y0, sWidth, sHeight), 
            plane={'C': c, 'Z': z, 'T': t}, 
            zoom=zoom,
            ).squeeze() 
        
        # Append sData
        sData['sImg'].append(sImg)
        sData['x0'].append(int(x0 * zoom))
        sData['y0'].append(int(y0 * zoom))
        sData['sWidth'].append(int(sWidth * zoom))
        sData['sHeight'].append(int(sHeight * zoom))
        sData['wID'].append(wID)
        sData['wTile'].append(wTile)
        
    # Extract relevant variables 
    sWidth = sData['sWidth'][0]
    sHeight = sData['sHeight'][0]
    x0 = np.stack([x0 for x0 in sData['x0']]) 
    y0 = np.stack([y0 for y0 in sData['y0']]) 
    xOr = np.min(x0); x0 -= xOr
    yOr = np.min(y0); y0 -= yOr
    xMax = np.max(x0) + sWidth
    yMax = np.max(y0) + sHeight
    
    # Make raw display
    dispRaw = np.zeros((yMax, xMax), dtype='int') 
    for s in range(len(sData['sImg'])):
        dispRaw[
            y0[s]:y0[s] + sHeight,
            x0[s]:x0[s] + sWidth
            ] = sData['sImg'][s]  
    
    # Get sorted unique well IDs (wID_unique)   
    def sort_key(value):
        match = re.match(r'(\D+)(\d+)', value)
        if match:
            prefix = match.group(1)
            number = int(match.group(2))
            return prefix, number
        return value
    wID = [wID for wID in sData['wID']]    
    wID_unique = list(set(wID))
    wID_unique = sorted(wID_unique, key=sort_key)
    
    # Count well rows and columns
    def count_letters_digits(strings_list):
        letters_set = set()
        digits_set = set()        
        for string in strings_list:
            letters = [char for char in string if char.isalpha()]
            digits = [char for char in string if char.isdigit()]
            letters_set.update(letters)
            digits_set.update(digits)
        wRow = len(letters_set)
        wCol = len(digits_set)
        return wRow, wCol
    wRow, wCol = count_letters_digits(wID_unique)   
        
    # Extract well data (wData)
    
    wData = {
        'wImg': [],
        'wImg_pad': [],
        'wWidth': [],
        'wHeight': [],
        'wID': [],
        }

    for wID in wID_unique:
        
        # Well info & image
        idx = [i for i, idx in enumerate(sData['wID']) if idx == wID]
        wsData = {key: value[idx[0]:idx[-1]+1] for key, value in sData.items()}
        x0 = np.stack([x0 for x0 in wsData['x0']]) 
        y0 = np.stack([y0 for y0 in wsData['y0']]) 
        xMin = np.min(x0) - xOr
        xMax = np.max(x0) - xOr + sWidth
        yMin = np.min(y0) - yOr
        yMax = np.max(y0) - yOr + sHeight 
        wWidth = xMax - xMin
        wHeight = yMax - yMin
        wImg = dispRaw[yMin:yMax, xMin:xMax]
        
        if noGap:
            nonzero_rows = np.where(np.any(wImg, axis=1))[0]
            nonzero_cols = np.where(np.any(wImg, axis=0))[0]
            wImg = wImg[np.ix_(nonzero_rows, nonzero_cols)]
            wWidth = wImg.shape[1]
            wHeight = wImg.shape[0]

        # Append wData
        wData['wImg'].append(wImg)
        wData['wWidth'].append(wWidth)
        wData['wHeight'].append(wHeight)
        wData['wID'].append(wID)
        
    # Pad well image 
    maxWidth = np.max([wWidth for wWidth in wData['wWidth']])
    maxHeight = np.max([wHeight for wHeight in wData['wHeight']])
    for i, wImg in enumerate(wData['wImg']):
        
        targetWidth = maxWidth + pad
        targetHeight = maxHeight + pad
        padX = targetWidth - wImg.shape[1]
        padY = targetHeight - wImg.shape[0]
        
        if padX % 2 == 0: padX0 = padX//2; padX1 = padX//2
        else: padX0 = padX//2; padX1 = padX//2 + 1   
        if padY % 2 == 0: padY0 = padY//2; padY1 = padY//2
        else: padY0 = padY//2; padY1 = padY//2 + 1  
                       
        wImg_pad = np.pad(wImg, (
            (padY0, padY1),
            (padX0, padX1)
            ), constant_values=7500)
        
        # Append wData
        wData['wImg_pad'].append(wImg_pad)
        
    # Make well display
    dispWell = np.zeros((wRow*wHeight, wCol*wWidth), dtype=int)
    dispWell = []
    for row in range(wRow):
        dispWell.append(np.hstack(wData['wImg_pad'][wCol*row:wCol*row+wCol]))
    dispWell = np.vstack(dispWell)
        
end = time.time()
print(f'  {(end-start):5.3f} s') 

#%%
    
io.imsave(
    img_path.replace('.czi', '_dispRaw.tif'),
    dispRaw.astype('uint16'),
    check_contrast=False,
    ) 

io.imsave(
    img_path.replace('.czi', '_dispWell.tif'),
    dispWell.astype('uint16'),
    check_contrast=False,
    ) 
