#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from pathlib import Path
from pylibCZIrw import czi as pyczi

#%% Parameters ----------------------------------------------------------------

zoom = 0.1
img_name = 'Digoxin_Bulfan_blm_mel5-01.czi'
img_path = str(Path('data') / img_name)

#%% Process -------------------------------------------------------------------

with pyczi.open_czi(img_path) as czidoc:

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
    md_chn = (md_all
              ['DisplaySetting']
              ['Channels']
              ['Channel']
             )
    
    # Get pixel size
    pixSize = float(md_all['Scaling']['Items']['Distance'][0]['Value'])
    pixSize *= 1e06 # convert to µm
    
    # Get number of channel
    nChannel = len(md_chn)
    
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
        
        for c in range(nChannel):
        
            # Scene image
            scene = czidoc.read(
                roi=(x0, y0, width, height), 
                plane={'C': c}, 
                zoom=zoom,
                ).squeeze()
