#%% Imports -------------------------------------------------------------------

# import csv
import csv
import time
import numpy as np
from skimage import io 
# from pathlib import Path
from pylibCZIrw import czi as pyczi
from skimage.draw import circle_perimeter
from skimage.feature import peak_local_max

#%% Initialize ----------------------------------------------------------------

scene_path = 'D:/ETH-ScopeM_CD7-GA/HU4NZLpR-202304051711.czi'
csv_path = 'C:/Users/bdeha/Projects/ETH-ScopeM_CD7-GA/xyStage.csv'

# Scene extraction
channel = 2
zoom = 0.1

# Local maxima detection
min_prominence = 1500
min_distance = 2

#%% Process -------------------------------------------------------------------

start = time.time()
print('Process')

with pyczi.open_czi(scene_path) as czidoc:

    # Read metadata
    md_xml = czidoc.raw_metadata
    md_all = (czidoc.metadata
              ['ImageDocument']
              ['Metadata']
              )
    md_scn = (md_all
              ['Information']
              ['Image']
              ['Dimensions']
              ['S']['Scenes']['Scene']
              )
    
    # Get pixel size
    pixSize = float(md_all['Scaling']['Items']['Distance'][0]['Value'])
    pixSize *= 1e06 # convert to µm
          
    # Extract scene data (sData)
    sData = {
        'scene': [],
        'wellID': [],
        'wellTile': [],
        'xstage': [],
        'ystage': [],
        'x0stage': [],
        'y0stage': [],
        'x0': [],
        'y0': [],
        'width': [],
        'height': [],
        'xLM': [],
        'yLM': [],
        'xLMStage': [],
        'yLMStage': [],
        }
    
    # Extract scene coordinates (sCoords)
    sCoords = czidoc.scenes_bounding_rectangle

    for s in range(len(sCoords)):
               
        # Scene coordinates (pix, top left)
        x0 = sCoords[s][0]
        y0 = sCoords[s][1]
        width = sCoords[s][2]
        height = sCoords[s][3]
        
        # Scene metadata
        wellID = md_scn[s]['ArrayName']
        wellTile = md_scn[s]['@Name']
        # Stage position (µm, mid)
        stage = md_scn[s]['CenterPosition']
        xstage = float(stage[0:stage.index(',')])
        ystage = float(stage[stage.index(',')+1:-1])
        # Stage position (µm, top left)
        x0stage = xstage - (width * 0.5 * pixSize)
        y0stage = ystage - (height * 0.5 * pixSize)
        
        # Scene image
        scene = czidoc.read(
            roi=(x0, y0, width, height), 
            plane={'C': channel}, 
            zoom=zoom,
            ).squeeze()
        
        # Local maxima
        locMax = peak_local_max(
            scene, min_distance=min_distance, threshold_abs=min_prominence
            )
        xLM = locMax[:,0] / zoom
        yLM = locMax[:,1] / zoom
        xLMStage = (xLM * pixSize) + x0stage
        yLMStage = (yLM * pixSize) + y0stage

        # Append sData
        sData['scene'].append(scene)
        sData['wellID'].append(wellID)
        sData['wellTile'].append(wellTile)
        sData['xstage'].append(xstage)
        sData['ystage'].append(ystage)
        sData['x0stage'].append(x0stage)
        sData['y0stage'].append(y0stage)
        sData['x0'].append(x0)
        sData['y0'].append(y0)
        sData['width'].append(width)
        sData['height'].append(height)
        sData['xLM'].append(xLM)
        sData['yLM'].append(yLM)
        sData['xLMStage'].append(xLMStage)
        sData['yLMStage'].append(yLMStage)

end = time.time()
print(f'  {(end-start):5.3f} s') 
  
#%% Save ----------------------------------------------------------------------

# Extract xLMStage and yLMStage in a single ndarray
xyStage = np.column_stack((
    np.concatenate(sData['xLMStage'], axis=0),
    np.concatenate(sData['yLMStage'], axis=0)
    ))

# Get a random subset of the ndarray
n = 100
np.random.seed(42)
indices = np.random.choice(xyStage.shape[0], size=n, replace=False)
xyStage_subset = xyStage[indices, :]

# Save the subset as csv
np.savetxt(csv_path, xyStage_subset, delimiter=',', fmt='%.2f')

#%% Display --------------------------------------------------------------------

# start = time.time()
# print('Display')

# # Extract all scenes (from sData)
# allScenes = np.stack(
#     [sData['scene'][i] for i in range(len(sData['scene']))]
#     )

# io.imsave(
#     scene_path.replace('.czi', '_allScenes.tif'),
#     allScenes.astype('uint16'),
#     check_contrast=False,
#     ) 

# # Make a tiled display
# xSize = int(md_all['Information']['Image']['SizeX'])
# ySize = int(md_all['Information']['Image']['SizeY'])

# tileDisplay = np.zeros((int(ySize*zoom), int(xSize*zoom)))
# for i, scene in enumerate(allScenes):
    
#     x0 = int(sData['x0'][i]*zoom)
#     y0 = int(sData['y0'][i]*zoom)
#     width = int(sData['width'][i]*zoom)
#     height = int(sData['height'][i]*zoom)
#     tileDisplay[y0:y0+height,x0:x0+width] = scene
    
# io.imsave(
#     scene_path.replace('.czi', '_tileDisplay.tif'),
#     tileDisplay.astype('uint16'),
#     check_contrast=False,
#     ) 

# end = time.time()
# print(f'  {(end-start):5.3f} s') 

# io.imsave(
#     scene_path.replace('.czi', '_scenes.tif'),
#     scenes.astype('uint16'),
#     check_contrast=False,
#     ) 

# display = []
# for scene in scenes:
    
#     tmpDisplay = np.zeros_like(scene, dtype=('uint16'))
#     locMax = peak_local_max(
#         scene, min_distance=min_distance, threshold_abs=min_prominence
#         )

#     for coord in locMax:
#         x = coord[0]
#         y = coord[1]
#         rr, cc = circle_perimeter(x, y, 2)
#         tmpDisplay[rr, cc] = np.max(scene)

#     display.append(np.stack((tmpDisplay, scene), axis=0)) 

# display = np.stack(display)

# end = time.time()
# print(f'  {(end-start):5.3f} s') 

# io.imsave(
#     scene_path.replace('.czi', '_display.tif'),
#     display,
#     check_contrast=False,
#     imagej=True,
#     metadata={'axes': 'TCYX'}
#     )

