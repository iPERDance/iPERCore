# Installation

## System Dependencies
 - Linux (test on Ubuntu 16.04 and 18.04) or Windows (test on windows 10).
 - CUDA 10.1, 10.2, 11.0, or 11.1 with Nvidia GPU. Set the path of $CUDA_HOME, and add the $CUDA_HOME/bin into the system path. 
 - gcc in Linux (supports C++14 and tests on 7.5+ ) or MSVC++ (Visual Studio 2019, supports C++14) in Windows.
 - ffmpeg (ffprobe) test on 4.3.1+.
 - Git test on 2.16.2+
 
 
### Linux (test on Ubuntu 16.04 and 18.04)
 - Download the CUDA >= 10.1, and set the `CUDA_HOME` into the system environments. This is important.
 Since in the next setup stage, it needs to get the `CUDA_HOME`
 For example,
     ```shell
     export CUDA_HOME=/usr/local/cuda-10.1
     export PATH=${CUDA_HOME}/bin:$PATH
     export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH 
     ```
   
 - Make sure that the gcc support C++14. Since in the next setup stage, it needs to compile some c++ codes from sources.
 Here, we use gcc 7.5.
 
 - It needs ffmpeg. If you have not installed it yet, you can run the followings
    ```shell
   apt-get install ffmpeg 
   ```
 - It needs Git. If you have not installed it yet, you can run the followings
    ```shell
   apt-get install git 
   ```
   
### Windows 10
 - [Download and install](https://git-scm.com/download/win) Git. 
 
 - [Download and install](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) the CUDA >= 10.1;
 
 - [Download](http://101.32.75.151:20086/iper_plus_plus_latest_executables.zip) ffmpeg. Unzip the `executables.zip` and move
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

**If the GPU device is RTX30+ (RTX 3060, RTX 3060ti, RTX 3070, RTX 3080, RTX 3090 ...), run the follows **
```shell
torch=1.8.1 python setup.py develop
```

It might takes 5-10 min. When you see `Finished processing dependencies for iPERCore==0.1.1` in the console screen, it means
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

**If the above links are broken, try to download the "assets/checkpoints" and "assets/samples.zip" from the following available links:**
* [OneDrive 1](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/liandz_shanghaitech_edu_cn/ErkIzzi6n0RLrP9gP5k2tpcB2BRzeRMok9moOgEUnpqX8A?e=Pq1omh)

* [OneDrive 2](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/liuwen_shanghaitech_edu_cn/EiOrbTo4yUtBrgQ4KiKswxUB-UyYl69W-pSVMNeFcXwYIw?e=z1Fyea)

* [OneDrive 3](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/qianych_shanghaitech_edu_cn/Evg7YNYDV5xGjox7qIsUe1IBvh3vodNPY1-1x4JpfX1bcQ?e=oW6Qcn)

* [BaiduPan](https://pan.baidu.com/s/1zEpPaU505Df13LOyF-H3Pw), password: `uomm`

Unzip **checkpoints.zip** and **samples.zip**, and move them to the `./assets` directory.






