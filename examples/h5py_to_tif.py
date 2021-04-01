import h5py
import numpy as np

import utils2p

def h5py_to_tif(h5py_in_path, tif_out_path=None):
    stack = []
    with h5py.File(h5py_in_path, "r") as file_handle:
        stack = file_handle.get("data")
        stack = np.squeeze(stack)
    if tif_out_path is not None:
        utils2p.save_img(tif_out_path, stack)
    return stack

def merge_left_right(left, right, x_size_out):
    assert left.shape == right.shape

    N_frames, size_y, size_x_small = left.shape
    overlap = 2 * size_x_small - x_size_out
    stack_out = np.zeros((N_frames, size_y, x_size_out), dtype=left.dtype)

    interpolate_factor = np.ones_like(left)
    interpolate_factor[:, :, -overlap:] = np.linspace(1,0, overlap)
    stack_out[:, :, :size_x_small] += left * interpolate_factor

    interpolate_factor = np.ones_like(right)
    interpolate_factor[:, :, :overlap] = np.linspace(0,1, overlap)
    stack_out[:, :, -size_x_small:] += right * interpolate_factor

    return stack_out

def double_h5py_to_tif(h5py_left_path, h5py_right_path, x_size_out, tif_out_path=None):
    left = h5py_to_tif(h5py_left_path)
    right = h5py_to_tif(h5py_right_path)

    whole_stack = merge_left_right(left, right, x_size_out)
    if tif_out_path is not None:
        utils2p.save_img(tif_out_path, whole_stack)
    return whole_stack

if __name__ == "__main__":
    # a = np.zeros((100, 320, 320))
    # b = np.ones((100, 320, 320))
    # c = merge_left_right(a, b, x_size_out=576)
    stack = h5py_to_tif("/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_12_19_57_2021_03_12_19_57/inference_halves/longterm_003_crop_out_right.h5",
                        "/home/jbraun/bin/deepinterpolation/sample_data/denoised_right_longterm_003_crop_out.tif")
    stack = h5py_to_tif("/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_12_19_57_2021_03_12_19_57/inference_halves/longterm_003_crop_out_left.h5",
                        "/home/jbraun/bin/deepinterpolation/sample_data/denoised_left_longterm_003_crop_out.tif")
    # stack = h5py_to_tif("/home/jbraun/bin/deepinterpolation/runs/longterm_unet_single_1024_mean_absolute_error_2021_03_12_19_57_2021_03_12_19_57/inference/longterm_003_crop_out.h5",
    #                     "/home/jbraun/bin/deepinterpolation/sample_data/denoised_longterm_003_crop_out.tif")
    # stack = h5py_to_tif("/home/jbraun/bin/deepinterpolation/runs/unet_single_1024_mean_absolute_error_2021_03_15_12_55_2021_03_15_12_55/inference/ophys_out.h5",
    #                     "/home/jbraun/bin/deepinterpolation/runs/unet_single_1024_mean_absolute_error_2021_03_15_12_55_2021_03_15_12_55/inference/ophys_out.tif")
    a = 0