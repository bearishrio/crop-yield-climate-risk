"""
download_climate.py
Downloads IMD gridded rainfall and temperature data for India.
"""

import imdlib as imd

START_YEAR = 1990
END_YEAR = 2019
SAVE_DIR = "yield/raw/climate/"

print("Downloading rainfall data... this may take a few minutes.")
imd.get_data('rain', START_YEAR, END_YEAR, fn_format='yearwise', file_dir=SAVE_DIR)

print("Downloading max temperature data...")
imd.get_data('tmax', START_YEAR, END_YEAR, fn_format='yearwise', file_dir=SAVE_DIR)

print("Downloading min temperature data...")
imd.get_data('tmin', START_YEAR, END_YEAR, fn_format='yearwise', file_dir=SAVE_DIR)

print("Done. Files saved to:", SAVE_DIR)