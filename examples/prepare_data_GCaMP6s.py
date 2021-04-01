# Jonas Braun
# 12.03.2021
# jonas.braun@epfl.ch
import numpy as np

import utils2p

raw_data_paths = ["/mnt/labserver/AYMANNS_Florian/Experimental_data/201014_G23xU1/Fly1/005_coronal/2p/warped_green.tif",
                  "/home/jbraun/data/181220_Rpr_R57C10_GC6s_tdTom/Fly5/002_coronal/processed/green_aligned_warped.tif"
                  ]
data_out_paths = ["/home/jbraun/bin/deepinterpolation/sample_data/ABO_GCaMP6s_16Hz_crop.tif",
                  "/home/jbraun/bin/deepinterpolation/sample_data/R57C10_GCaMP6s_8Hz_crop.tif"
                  ]
size_y = 320  # 320
# overlap = -64  # 0
# size_x = size_y * 2 - overlap
size_x = 640

for i_trial, (raw_data_path, data_out_path) in enumerate(zip(raw_data_paths, data_out_paths)):
    stack = utils2p.load_img(raw_data_path)
    N_frames, N_y, N_x = stack.shape

    if i_trial == 0:
        size_x = 672

    offset_y = np.floor((N_y-size_y) / 2).astype(np.int)
    offset_x = np.floor((N_x-size_x) / 2).astype(np.int)

    stack_crop = stack[:, offset_y:offset_y+size_y, offset_x:offset_x+size_x]

    utils2p.save_img(data_out_path, stack_crop)

pass