# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
from os.path import exists
import platform
from setuptools import find_packages, setup
import subprocess
import sys
import re

TORCH_DIST = "https://download.pytorch.org/whl/torch_stable.html"
MMCV_DIST = "https://download.openmmlab.com/mmcv/dist"

PIP_VERSION = "20.2.4"


#                        cuda
# torch 1.8.1       +cu101     +cu102      +cu111
# torch 1.8.0       +cu101      cu102      +cu111
# torch 1.7.1       +cu101      cu102      +cu110
# torch 1.7.0       +cu101      cu102      +cu110
# torch 1.6.0       +cu101      cu102

VALID_CUDA = [10.1, 10.2, 11.0, 11.1]

DEFAULT_LINUX_CUDA_TORCH = {
    "cu111": "1.8.1",
    "cu110": "1.7.0",
    "cu102": "1.7.0",
    "cu101": "1.7.0"
}

DEFAULT_WINDOWS_CUDA_TORCH = {
    "cu102": "1.6.0",
    "cu101": "1.6.0"
}


PRECOMPILED_TORCH_CUDA_PAIRS = {
    "1.8.1+cu111": {
        "torch": "1.8.1+cu111",
        "torchvision": "0.9.1+cu111",
        "mmcv-full": "1.2.0",
        "mmcv-dist": f"{MMCV_DIST}/cu111/torch1.8.1/index.html"
    },
    "1.8.1+cu102": {
        "torch": "1.8.1+cu102",
        "torchvision": "0.9.1+cu102",
        "mmcv-full": "1.2.0",
        "mmcv-dist": f"{MMCV_DIST}/cu102/torch1.8.1/index.html"
    },
    "1.8.1+cu101": {
        "torch": "1.8.1+cu101",
        "torchvision": "0.9.1+cu101",
        "mmcv-full": "1.2.0",
        "mmcv-dist": f"{MMCV_DIST}/cu101/torch1.8.1/index.html"
    },
    "1.7.0+cu110": {
        "torch": "1.7.0+cu110",
        "torchvision": "0.8.1+cu110",
        "mmcv-full": "1.2.0",
        "mmcv-dist": f"{MMCV_DIST}/cu110/torch1.7.0/index.html"
        # "mmcv-full": "1.2.0+torch1.7.0+cu110"
    },
    "1.7.0+cu102": {
        "torch": "1.7.0",
        "torchvision": "0.8.1",
        "mmcv-full": "1.2.0",
        "mmcv-dist": f"{MMCV_DIST}/cu102/torch1.7.0/index.html"
        # "mmcv-full": "1.2.0+torch1.7.0+cu102"
    },
    "1.7.0+cu101": {
        "torch": "1.7.0+cu101",
        "torchvision": "0.8.1+cu101",
        "mmcv-full": "1.2.0",
        "mmcv-dist": f"{MMCV_DIST}/cu101/torch1.7.0/index.html"
        # "mmcv-full": "1.2.0+torch1.7.0+cu101"
    },
    "1.6.0+cu102": {
        "torch": "1.6.0",
        "torchvision": "0.7.0",
        "mmcv-full": "1.2.0",
        "mmcv-dist": f"{MMCV_DIST}/cu102/torch1.6.0/index.html"
        # "mmcv-full": "1.1.5+torch1.6.0+cu102"
    },
    "1.6.0+cu101": {
        "torch": "1.6.0+cu101",
        "torchvision": "0.7.0+cu101",
        "mmcv-full": "1.2.0",
        "mmcv-dist": f"{MMCV_DIST}/cu101/torch1.6.0/index.html"
        # "mmcv-full": "1.1.5+torch1.6.0+cu101"
    }
}

WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS = {
    "1.6.0+cu102": {
        "torch": "https://download.pytorch.org/whl/cu102/torch-1.6.0-cp{PYTHON_VERSION}-cp{PYTHON_ABI_VERSION}-win_amd64.whl",
        "torchvision": "https://download.pytorch.org/whl/cu102/torchvision-0.7.0-cp{PYTHON_VERSION}-cp{PYTHON_ABI_VERSION}-win_amd64.whl",
        "mmcv-full": "1.1.5",
        "mmcv-dist": f"{MMCV_DIST}/cu102/torch1.6.0/index.html"
    },
    "1.6.0+cu101": {
        "torch": "https://download.pytorch.org/whl/cu101/torch-1.6.0%2Bcu101-cp{PYTHON_VERSION}-cp{PYTHON_ABI_VERSION}-win_amd64.whl",
        "torchvision": "https://download.pytorch.org/whl/cu101/torchvision-0.7.0%2Bcu101-cp{PYTHON_VERSION}-cp{PYTHON_ABI_VERSION}-win_amd64.whl",
        "mmcv-full": "1.1.5",
        "mmcv-dist": f"{MMCV_DIST}/cu101/torch1.6.0/index.html"
    }
}


def get_cuda_version() -> float:
    """
        Get the cuda version of the system, make sure that the $CUDA_HOME or the $CUDA_PATH has been added into the system path.
        Here the cuda version is aaa.b.ccc, and only return the version with aaa.b
    Returns:
        version (float): the cuda version.

    """
    global VALID_CUDA

    # In windows, the cuda_install.exe will set the `CUDA_PATH` to the system environmental variables.
    if "CUDA_HOME" not in os.environ and "CUDA_PATH" in os.environ:
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"]

    assert "CUDA_HOME" in os.environ, r"Cannot find the $CUDA_HOME in the environments. Please manually install the " \
                                      r"CUDA >= 10.1, and set the $CUDA_HOME environment variable."

    cuda_version_file = os.path.join(os.environ["CUDA_HOME"], "version.txt")

    if os.path.exists(cuda_version_file):
        # e.g. "CUDA Version 10.1.243", "CUDA Version 10.0.130"
        with open(cuda_version_file) as f:
            version_str = f.readline().replace("\n", "").replace("\r", "")

        # "CUDA Version 10.1.243" -> ["CUDA", "Version", "10.1.243"] -> "10.1.243"
        version = version_str.split(" ")[2]

        # "10.1.243" -> "10.1" -> 10.1
        version = float(".".join(version.split(".")[0:2]))

    else:
        # run `nvcc -V`
        # """nvcc: NVIDIA (R) Cuda compiler driver
        #    Copyright (c) 2005-2019 NVIDIA Corporation
        #    Built on Sun_Jul_28_19:07:16_PDT_2019
        #    Cuda compilation tools, release 10.1, V10.1.243
        # """

        nvcc_out = subprocess.run("nvcc -V", shell=True, stdout=subprocess.PIPE)
        nvcc_str = nvcc_out.stdout.decode("utf-8")
        nvcc_cuda = re.findall(r"[.]*([\d]+.[\d]+),[.]*", nvcc_str)

        if len(nvcc_cuda) == 0:
            raise RuntimeError(f"nvcc -V error! {nvcc_str}")
        else:
            version = float(nvcc_cuda[0])

    assert version in VALID_CUDA, f"CUDA Version {version} must in {VALID_CUDA}. " \
                                  f"Please manually install the CUDA meets the requirements."

    print(f"Cuda version is {version}")

    return version


def get_python_version() -> str:
    """
        Get the python version. The python version is aaa.b.c, and it only returns aaa.b
    Returns:
        version (str):
    """

    version = str(platform.python_version())
    version = "".join(version.split(".")[0:2])

    assert "36" <= version <= "38", f"Currently, it only support the python version with 3.6.+, 3.7.+, 3.8.+"

    return version


def get_torch_version(cuda_version_str, precompile_torch_cuda_paris, default_torch_cuda_mapper) -> str:
    """

    Args:
        cuda_version_str:
        precompile_torch_cuda_paris (dict): PRECOMPILED_TORCH_CUDA_PAIRS or WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS
        default_torch_cuda_mapper (dict): DEFAULT_LINUX_CUDA_TORCH or DEFAULT_WINDOWS_CUDA_TORCH

    Returns:
        torch_version (str):
    """

    if "torch" in os.environ:
        attempt_torch_version_str = os.environ["torch"]
        torch_cuda_str = f"{attempt_torch_version_str}+{cuda_version_str}"

        if torch_cuda_str not in precompile_torch_cuda_paris:
            torch_version_str = default_torch_cuda_mapper[cuda_version_str]
            print(f"torch=={attempt_torch_version_str} is incompatible with cuda {cuda_version_str}. "
                  f"The compatible torch-cuda version are {precompile_torch_cuda_paris.keys()},"
                  f"and here we install the torch+cuda=={torch_version_str} + {cuda_version_str}.")

        else:
            torch_version_str = attempt_torch_version_str

    else:
        torch_version_str = default_torch_cuda_mapper[cuda_version_str]

    return torch_version_str


def platform_dependencies():
    """Parse the pre-complied consistent versions of torch, torchvision, mmcv, and CUDA.
    The torch version must >= 1.6.0, and the CUDA version must >= 10.1.
    Currently, it only supports Linux and Windows.
    If the platform is Linux, we will use torch 1.7.0 + CUDA Version.
    Otherwise if the platform is windows, we will use torch 1.6.0 + CUDA VERSION.

    Returns:
        List[List[str]]: list of setup requirements items.

    """
    global TORCH_DIST, PRECOMPILED_TORCH_CUDA_PAIRS, WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS, \
        DEFAULT_LINUX_CUDA_TORCH, DEFAULT_WINDOWS_CUDA_TORCH

    cuda_version = get_cuda_version()
    cuda_version_str = "cu" + str(cuda_version).replace(".", "")

    packages = []

    if platform.system().lower() == "windows":
        python_version = get_python_version()
        python_abi_version = python_version
        if python_version != "38":
            python_abi_version += "m"

        torch = get_torch_version(cuda_version_str, WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS, DEFAULT_WINDOWS_CUDA_TORCH)
        torch_cuda_version = f"{torch}+{cuda_version_str}"
        numpy_version = "numpy==1.19.3"

        assert torch_cuda_version in WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS, \
            f"There is no pre-complied pytorch 1.6.0 with CUDA {cuda_version}, " \
            f"and you might need to install pytorch 1.6.0 with CUDA {cuda_version} from source."

        torch_link = WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["torch"] \
            .format(PYTHON_VERSION=python_version, PYTHON_ABI_VERSION=python_abi_version)
        torchvision_link = WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["torchvision"] \
            .format(PYTHON_VERSION=python_version, PYTHON_ABI_VERSION=python_abi_version)

        mmcv_version = WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["mmcv-full"]
        mmcv_dist = WINDOWS_PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["mmcv-dist"]

        packages.append([f"{torch_link}", "-f", TORCH_DIST])
        packages.append([f"{torchvision_link}", "-f", TORCH_DIST])
        packages.append([f"mmcv-full=={mmcv_version}", "-f", mmcv_dist])
        packages.append(numpy_version)

    elif platform.system().lower() == "linux":
        torch = get_torch_version(cuda_version_str, PRECOMPILED_TORCH_CUDA_PAIRS, DEFAULT_LINUX_CUDA_TORCH)
        torch_cuda_version = f"{torch}+{cuda_version_str}"
        numpy_version = "numpy>=1.19.3"

        assert torch_cuda_version in PRECOMPILED_TORCH_CUDA_PAIRS, \
            f"There is no pre-complied pytorch {torch} with CUDA {cuda_version} in {TORCH_DIST}, " \
            f"and you might need to install pytorch {torch} with CUDA {cuda_version} from source."

        torch_version = PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["torch"]
        torchvision_version = PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["torchvision"]
        mmcv_version = PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["mmcv-full"]
        mmcv_dist = PRECOMPILED_TORCH_CUDA_PAIRS[torch_cuda_version]["mmcv-dist"]

        packages.append([f"torch=={torch_version}", "-f", TORCH_DIST])
        packages.append([f"torchvision=={torchvision_version}", "-f", TORCH_DIST])
        packages.append([f"mmcv-full=={mmcv_version}", "-f", mmcv_dist])
        packages.append(numpy_version)

    else:
        raise ValueError(f"Currently it only supports 'windows' and 'linux'.")

    return packages


def parse_requirements(fname="requirements.txt", with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.
    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs
    Returns:
        List[str]: list of requirements items
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                info["package"] = line
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest  # NOQA
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())

    return packages


# 1. install torch, torchvision, and mmcv firstly.
torch_torchvision_mmcv = platform_dependencies()

# 2. other installed requires
install_requires = parse_requirements("requirements/runtime.txt")
# 3. build requires
build_requiers = parse_requirements("requirements/build.txt")

# 4. pip install all of them
all_requires = [[f"pip=={PIP_VERSION}"]] + torch_torchvision_mmcv + install_requires + build_requiers

pip_executable = [sys.executable, "-m", "pip", "install"]

for package_line in all_requires:
    if isinstance(package_line, str):
        package_line = [package_line]

    pip_install_line = pip_executable + package_line

    print(" ".join(pip_install_line))
    subprocess.run(pip_install_line)

# 5. setup iPERCore
setup(
    name="iPERCore",
    version="0.2.0",
    author="Wen Liu, and Zhixin Piao",
    author_email="liuwen@shanghaitech.edu.cn",
    url="https://github.com/iPERDance/iPERCore",
    description="The core of impersonator++.",
    packages=find_packages(exclude=("assets",)),
    python_requires=">=3.6, <=3.8",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "run_imitator = iPERCore.services.run_imitator:run_imitator",
            "run_swapper = iPERCore.services.run_imitator:run_swapper",
            "run_viewer = iPERCore.services.run_imitator:run_viewer",
        ]
    }
)
