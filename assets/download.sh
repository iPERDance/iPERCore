#! /bin/bash

# download checkpoints
wget -O assets/checkpoints.zip "https://download.impersonator.org/iper_plus_plus_latest_checkpoints.zip"
unzip -o assets/checkpoints.zip  -d  assets
rm assets/checkpoints.zip


# download samples
wget -O assets/samples.zip "https://download.impersonator.org/iper_plus_plus_latest_samples.zip"
unzip -o assets/samples.zip -d  assets
rm assets/samples.zip

## download executable files, such as ffmpeg. Windows platform needs this.
## wget -O assets/executables.zip  "https://download.impersonator.org/iper_plus_plus_latest_executables.zip"
# wget -O assets/executables.zip  "http://101.32.75.151:20086/iper_plus_plus_latest_executables.zip"
# unzip -o assets/executables.zip  -d assets
# rm assets/executables