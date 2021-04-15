
# iPERCore-0.1
- all base codes, and the motion imitation;

# iPERCore-0.1.1
- Fix the some bugs in installation. Directly down the link whi of Windows version of Torch, TorchVision, and MMCV-full.
- the permission of os.symbolic in Windows. Use shutil.copy instead of os.symbolic. Though, copy files might not efficient than symbolic.
- fix the multi-preprocessing problem in Windows.

# iPERCore-0.2.0

- Fix the error parts annotations in [smpl_part_info.json](../assets/configs/pose3d/smpl_part_info.json). We will delete
  the original file *smpl_part_info.json* in *assets/configs/pose3d/smpl_part_info.json*. 