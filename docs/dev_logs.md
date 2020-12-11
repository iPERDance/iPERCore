
# iPERCore-0.1
- all base codes, and the motion imitation;

# iPERCore-0.1.1
- Fix the some bugs in installation. Directly down the link whi of Windows version of Torch, TorchVision, and MMCV-full.
- the permission of os.symbolic in Windows. Use shutil.copy instead of os.symbolic. Though, copy files might not efficient than symbolic.
- fix the multi-preprocessing problem in Windows.