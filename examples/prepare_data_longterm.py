# Jonas Braun
# 12.03.2021
# jonas.braun@epfl.ch
import numpy as np

import utils2p

raw_data_paths = [# "/home/jbraun/data/longterm/210212/Fly1/cs_003/processed/green_com_warped.tif",
                  "/home/jbraun/data/longterm/210212/Fly1/cs_005/processed/green_com_warped.tif",
                  "/home/jbraun/data/longterm/210212/Fly1/cs_007/processed/green_com_warped.tif",
                  "/home/jbraun/data/longterm/210212/Fly1/cs_010/processed/green_com_warped.tif",
                  ]
data_out_paths = [# "/home/jbraun/bin/deepinterpolation/sample_data/longterm_003_crop.tif",
                  "/home/jbraun/bin/deepinterpolation/sample_data/longterm_005_crop.tif",
                  "/home/jbraun/bin/deepinterpolation/sample_data/longterm_007_crop.tif",
                  "/home/jbraun/bin/deepinterpolation/sample_data/longterm_010_crop.tif",
                  ]
size_y = 320
overlap = 64
size_x = size_y * 2 - overlap

for i_trial, (raw_data_path, data_out_path) in enumerate(zip(raw_data_paths, data_out_paths)):
    stack = utils2p.load_img(raw_data_path)
    N_frames, N_y, N_x = stack.shape

    offset_y = np.floor((N_y-size_y) / 2).astype(np.int)
    offset_x = np.floor((N_x-size_x) / 2).astype(np.int)

    stack_crop = stack[:, offset_y:offset_y+size_y, offset_x:offset_x+size_x]

    utils2p.save_img(data_out_path, stack_crop)

pass