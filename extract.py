#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from pathlib import Path
from itertools import product
from pylibCZIrw import czi as pyczi
from joblib import Parallel, delayed 

#%% Comments ------------------------------------------------------------------

'check variable name consistency'

#%% Names ---------------------------------------------------------------------

# czi_name = 'Masina_CD7_(3c-540s)_20x_16bits.czi'
# czi_name = 'Stebler_CD7_(4c-120s)_5x-2x_14bits.czi'
# czi_name = 'Stoma_CD7_(3z-4c-240s)_5x-2x_16bits.czi'
# czi_name = 'Bertet_880_(100t-10z)_40x_8bits.czi'
# czi_name = 'Lelouard_780_(14z-6c)_40x_8bits.czi'
czi_name = 'Sidor_880_(6z-4c)_100x_16bits.czi'
# czi_name = 'Bruneau_Z2_(11z-3c)_20x_8bits_Stitching.czi'
# czi_name = 'Aggad_880_(5z-3c)_40x_8bits.czi'

#%% Initialize ----------------------------------------------------------------

T = 'all'
Z = 'all'
C = 'all'

zoom = 0.1

czi_path = str(Path('data') / czi_name)

#%% Process -------------------------------------------------------------------

start = time.time()
print('Process')

# Read metadata
with pyczi.open_czi(czi_path) as czidoc:    
    md_all = czidoc.metadata['ImageDocument']['Metadata']
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

# Read channel info
chn_name = []
for chn in range(nC):
    if nC <= 1: chn_name.append(md_chn['@Name'])
    else: chn_name.append(md_chn[chn]['@Name'])
chn_name = tuple(chn_name)

# Read pixel info
pix_size = []
for pix in md_pix:
    pix_size.append(pix['Value'])
pix_size = tuple(pix_size)    
# pix_size = float(md_all['Scaling']['Items']['Distance'][0]['Value'])
# pixType = md_img['PixelType']
   
# Append metadata dict      
metadata = {    
    'nT': nT, 'nZ': nZ, 'nC': nC, 'nY': nY, 'nX': nX, 'nS': nS, 
    'pix_size': pix_size, # 'pixType': pixType,
    'chn_name': chn_name,
    }
