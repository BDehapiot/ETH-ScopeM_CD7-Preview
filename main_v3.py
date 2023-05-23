#%% Imports -------------------------------------------------------------------

import re
import time
import numpy as np
from skimage import io 
from pathlib import Path
from itertools import product
from pylibCZIrw import czi as pyczi
from joblib import Parallel, delayed 

#%% Parameters ----------------------------------------------------------------

T = 0
Z = 'all'
C = [0,1]

zoom = 0.05
pad = 20 
noGap = True

img_name = 'Digoxin_Bulfan_blm_mel5-01.czi'
# img_name = 'test02-withsoftwareAF-20x1x-095-60wells-01.czi'
# img_name = 'HU4NZLpR-202304051711.czi'
img_path = str(Path('data') / img_name)

#%% Initialize ----------------------------------------------------------------

start = time.time()
print('Initialize')

with pyczi.open_czi(img_path) as czidoc:
    
    # Read metadata
    md_all = czidoc.metadata['ImageDocument']['Metadata']
    md_img = (md_all['Information']['Image'])
    md_scn = (md_img['Dimensions']['S']['Scenes']['Scene'])
    
    # Extract scene coordinates
    sCoords = czidoc.scenes_bounding_rectangle
    
# Extract dimensions
nX = int(md_img['SizeX'])
nY = int(md_img['SizeY'])
nC = int(md_img['SizeC'])
nZ = int(md_img['SizeZ'])
nT = int(md_img['SizeT'])
nS = int(md_img['SizeS'])
    
# Determine extraction patterns                      
def handle_dims(dim, nDim):
    if dim == 'all':
        return np.arange(nDim)
    elif isinstance(dim, list):
        return np.array(dim)
    elif isinstance(dim, int):
        return np.array([dim])       
t = handle_dims(T, nT)
z = handle_dims(Z, nZ)
c = handle_dims(C, nC)
patterns = list(product(t, z, c))
patterns = np.array(patterns)
idxs = np.empty_like(patterns)
for i in range(patterns.shape[1]):
    _, inverse = np.unique(patterns[:, i], return_inverse=True)
    idxs[:, i] = inverse    
    
end = time.time()
print(f'  {(end-start):5.3f} s') 
    
#%% Extract scenes ------------------------------------------------------------
    
start = time.time()
print('Extract scenes')

# Create empty dict (sData)
sData = {
    'sImg': [],
    'x0': [],
    'y0': [],
    'sWidth': [],
    'sHeight': [],
    'wID': [],
    'wTile': [],
}  

def extract_scenes(s):
    
    # Scene info (x0 & y0 = top left pixel)
    x0, y0, sWidth, sHeight = sCoords[s][:4]
    wID = md_scn[s]['ArrayName']
    wTile = md_scn[s]['@Name']
            
    # Create empty image (sImg)
    sImg = np.zeros((
        t.size, z.size, c.size, 
        int(sHeight * zoom), 
        int(sWidth * zoom)
        ), dtype=int)
    
    # Extract scenes
    with pyczi.open_czi(img_path) as czidoc:        
        for pattern, idx in zip(patterns, idxs):
            img = czidoc.read(
                roi=(x0, y0, sWidth, sHeight), 
                plane={'T': pattern[0], 'Z': pattern[1], 'C': pattern[2]}, 
                zoom=zoom).squeeze()
            sImg[idx[0],idx[1],idx[2],...] = img
        
    # Scale coordinates (acc. to zoom)
    x0 = int(x0 * zoom)
    y0 = int(y0 * zoom)
    sWidth = int(sWidth * zoom)
    sHeight = int(sHeight * zoom)
        
    return sImg, x0, y0, sWidth, sHeight, wID, wTile

# Run extract_scenes in parallel
results = Parallel(n_jobs=-1)(
    delayed(extract_scenes)(s) 
    for s in range(len(sCoords))
    )

# Unpack and append results to sData
for result in results:
    sData['sImg'].append(result[0])
    sData['x0'].append(result[1])
    sData['y0'].append(result[2])
    sData['sWidth'].append(result[3])
    sData['sHeight'].append(result[4])
    sData['wID'].append(result[5])
    sData['wTile'].append(result[6])  
    
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
dispRaw = np.zeros((t.size, z.size, c.size, yMax, xMax), dtype=int)
for s in range(len(sData['sImg'])):
    dispRaw[...,
        y0[s]:y0[s] + sHeight,
        x0[s]:x0[s] + sWidth
        ] = sData['sImg'][s]
    
end = time.time()
print(f'  {(end-start):5.3f} s') 

#%% Extract wells -------------------------------------------------------------  
           
start = time.time()
print('Extract wells')
    
# Get sorted unique well IDs (wID_unique)   
def sort_wIDs(wIDs):
    match = re.match(r'(\D+)(\d+)', wIDs)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return prefix, number
    return wIDs
wIDs = [wID for wID in sData['wID']]    
wIDs_unique = list(set(wIDs))
wIDs_unique = sorted(wIDs_unique, key=sort_wIDs)

# Count well rows and columns
def count_rows_cols(strings_list):
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
wRow, wCol = count_rows_cols(wIDs_unique)   

# Create empty dict (wData)

wData = {
    'wImg': [],
    'wImg_pad': [],
    'wWidth': [],
    'wHeight': [],
    'wID': [],
    }

for wID in wIDs_unique:
    
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
    wImg = dispRaw[..., yMin:yMax, xMin:xMax]
       
    if noGap:
        wImg_flat = wImg.reshape((-1, wImg.shape[-2], wImg.shape[-1]))
        zeroRows = np.all(wImg_flat == 0, axis=1)
        zeroCols = np.all(wImg_flat == 0, axis=2)
        delRows = np.where(zeroRows)[0]
        delCols = np.where(zeroCols)[0]
        wImg = np.delete(wImg, delRows, axis=-2)
        wImg = np.delete(wImg, delCols, axis=-1)

    # if noGap:
        # nonzero_rows = np.where(np.any(wImg, axis=1))[0]
        # nonzero_cols = np.where(np.any(wImg, axis=0))[0]
        # wImg = wImg[np.ix_(nonzero_rows, nonzero_cols)]
        # wWidth = wImg.shape[1]
        # wHeight = wImg.shape[0]

    # Append wData
    wData['wImg'].append(wImg)
    wData['wWidth'].append(wWidth)
    wData['wHeight'].append(wHeight)
    wData['wID'].append(wID)
    
# Pad well image 
maxWidth = np.max([wWidth for wWidth in wData['wWidth']])
maxHeight = np.max([wHeight for wHeight in wData['wHeight']])
# for i, wImg in enumerate(wData['wImg']):
    
#     targetWidth = maxWidth + pad
#     targetHeight = maxHeight + pad
#     padX = targetWidth - wImg.shape[1]
#     padY = targetHeight - wImg.shape[0]
    
#     if padX % 2 == 0: padX0 = padX//2; padX1 = padX//2
#     else: padX0 = padX//2; padX1 = padX//2 + 1   
#     if padY % 2 == 0: padY0 = padY//2; padY1 = padY//2
#     else: padY0 = padY//2; padY1 = padY//2 + 1  
                   
#     wImg_pad = np.pad(wImg, (
#         (padY0, padY1),
#         (padX0, padX1)
#         ), constant_values=7500)
    
#     # Append wData
#     wData['wImg_pad'].append(wImg_pad)
    
# # Make well display
# dispWell = np.zeros((wRow*wHeight, wCol*wWidth), dtype=int)
# dispWell = []
# for row in range(wRow):
#     dispWell.append(np.hstack(wData['wImg_pad'][wCol*row:wCol*row+wCol]))
# dispWell = np.vstack(dispWell)
        
end = time.time()
print(f'  {(end-start):5.3f} s') 

#%%
    
io.imsave(
    img_path.replace('.czi', '_dispRaw.tif'),
    dispRaw.astype('uint16'),
    check_contrast=False,
    imagej=True,
    metadata={
        'axes': 'TZCYX', 
        }
    )

# io.imsave(
#     img_path.replace('.czi', '_dispRaw.tif'),
#     dispRaw.astype('uint16'),
#     check_contrast=False,
#     ) 

# io.imsave(
#     img_path.replace('.czi', '_dispWell.tif'),
#     dispWell.astype('uint16'),
#     check_contrast=False,
#     ) 
