# Installation

## System Dependencies
 - Linux (test on Ubuntu 16.04 and 18.04) or Windows (test on windows 10)
 - CUDA 10.1, 10.2, or 11.0 with Nvidia GPU.
 - gcc 7.5+ in Linux (needs to support C++14) or MSVC++ in Windows.
 - ffmpeg (ffprobe) test on 4.3.1+.
 
 
### Linux (test on Ubuntu 16.04 and 18.04)
 - Download the CUDA >= 10.1, and set the `CUDA_HOME` into the system environments. This is important.
 Since in the next setup stage, it needs to get the `CUDA_HOME`
 For example,
     ```shell
     export CUDA_HOME=/usr/local/cuda-10.1
     ```
   
 - Make sure that the gcc support C++14. Since in the next setup stage, it needs to compile some c++ codes from sources.
 Here, we use gcc 7.5.
 
 - It needs ffmpeg. If you do not install it. You can run the followings
    ```shell
   apt-get install ffmpeg 
   ```
   
### Windows 10
 - [Download and install](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) the CUDA >= 10.1;
 
 - [Download](https://1drv.ws/u/s!AjjUqiJZsj8whLkv8NeuckqVWz0H3A?e=a9ROXZ) ffmpeg. Unzip the `executables.zip` and move
 the `executables` into `./assets` folder.
 
## Python Dependencies

### Setup Python
Using Python 3.6+. You can use Anaconda Python 3.6+, or the native Python 3.6+.
Whatever Python you have used, be sure to create a virtual environment firstly.

 - For Anaconda Python
    ```shell
   conda create -n venv python=3.6.6
   conda activate venv
   
   ```

 
 - For Native Python
   ```shell
   pip install virtualenv
   virtualenv  venv
   source venv/bin/activate
   
   ```
   
### Requirements and Install iPERCore
Install iPERCore by running the follows

```shell
python setup.py develop
```

It might takes 5-10 min. When you see `Finished processing dependencies for iPERCore==0.1` in the console screen, it means
you have installed the `iPERCore` successfully.


The details of [requirements](../requirements/full_reqs.txt). Again be sure you have used a virtual environment to
avoid the conflicts with your own python site-packages.

# Download checkpoints and samples

Run the followings scripts to download checkpoints and samples (recommend).
```shell
sh assets/download.sh
```

Or manually download the checkpoints and samples (option).
- checkpoints.zip: https://download.impersonator.org/iper_plus_plus_latest_checkpoints.zip
- samples.zip: https://download.impersonator.org/iper_plus_plus_latest_samples.zip

Unzip all of them, and move them to the `./assets` directory.






