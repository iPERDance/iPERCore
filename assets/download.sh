#! /bin/bash

# download checkpoints
wget -O assets/checkpoints.zip "http://101.32.75.151:10086/checkpoints.zip"
unzip -o assets/checkpoints.zip  -d  assets
rm assets/checkpoints.zip


# download samples
wget -O assets/samples.zip "http://101.32.75.151:10086/samples.zip"
unzip -o assets/samples.zip -d  assets
rm assets/samples.zip

## download executable files, such as ffmpeg. Windows platform needs this.
# wget -O assets/executables.zip  "http://101.32.75.151:10086/executables.zip"
# unzip -o assets/executables.zip  -d assets
# rm assets/executables