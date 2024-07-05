#%% Imports -------------------------------------------------------------------

import time
from pathlib import Path
from functions import extract_metadata, extract_data, save_tiff, CD7_preview

#%% Inputs --------------------------------------------------------------------

data_path = 'D:\local_CZITools\data'
# czi_name = 'SpiroChem_5x_2x_plate-1_NA 39_20230818-01.czi'
# czi_name = 'Masina_CD7_(3c-540s)_20x_16bits.czi'
czi_name = 'Stebler_CD7_(4c-120s)_5x-2x_14bits.czi'
# czi_name = 'Stoma_CD7_(3z-4c-240s)_5x-2x_16bits.czi'
# czi_name = 'Bertet_880_(100t-10z)_40x_8bits.czi'
# czi_name = 'Bertet_880_(566t-15z)_40x_8bits.czi'
# czi_name = 'Lelouard_780_(14z-6c)_40x_8bits.czi'
# czi_name = 'Sidor_880_(6z-4c)_100x_16bits.czi'
# czi_name = 'Bruneau_Z2_(11z-3c)_20x_8bits_Stitching.czi'
# czi_name = 'Aggad_880_(5z-3c)_40x_8bits.czi'

#%% Initialize ----------------------------------------------------------------

czi_path = str(Path(data_path) / czi_name)

#%% Execute -------------------------------------------------------------------

start = time.time()
print('Extract metadata')

metadata = extract_metadata(czi_path)

end = time.time()
print(f'  {(end-start):5.3f} s') 

# -----------------------------------------------------------------------------
   
# start = time.time()
# print('Extract data')

# metadata, data = extract_data(czi_path, rT='all', rZ='all', rC='all', zoom=0.25)

# end = time.time()
# print(f'  {(end-start):5.3f} s') 

# -----------------------------------------------------------------------------

# start = time.time()
# print('Save tiff')

# save_tiff(czi_path, rT='all', rZ='all', rC='all', zoom=0.25, hyperstack=True)

# end = time.time()
# print(f'  {(end-start):5.3f} s') 

# -----------------------------------------------------------------------------

# start = time.time()
# print('CD7 preview')

# CD7_preview(czi_path, rT='all', rZ='all', rC='all', zoom=0.25,
#     uint8=False, labels=True, label_size=0.75, no_gap=True, pad_size=4
#     )

# end = time.time()
# print(f'  {(end-start):5.3f} s') 
