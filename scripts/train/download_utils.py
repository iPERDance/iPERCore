# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import os
import sys
import requests
from urllib.request import urlopen
from tqdm import tqdm


def download_from_url(url, dst):
    """

    Args:
        url (str):  url to download file
        dst (str):  dst place to put the file

    Returns:

    """
    file_size = int(urlopen(url).info().get("Content-Length", -1))

    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return True
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit="B", unit_scale=True, desc=url.split("/")[-1])
    req = requests.get(url, headers=header, stream=True)

    content_size = first_byte
    with(open(dst, "ab")) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if len(chunk) != 1024:
                print(len(chunk))

            content_size += len(chunk)
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()

    print(content_size, file_size)
    return content_size >= file_size


def robust_download_from_url(url, dst, attempt_times=10):
    """

    Args:
        url:
        dst:
        attempt_times:

    Returns:

    """

    success = False
    for i in range(0, attempt_times):
        print(f"Attempt to download from {url} ... ({i} < {attempt_times})")

        try:
            success = download_from_url(url, dst)
        except requests.exceptions.ConnectionError:
            print("ConnectionError and we will try to connect it again.")
        except requests.exceptions.ChunkedEncodingError:
            print("ChunkedEncodingError, and we will attempt to continue to download it.")
        except requests.exceptions.RequestException:
            print("An Unknown Error Happened, and we will attempt to continue to download it.")

        if success:
            break

    return success


def download_from_url_to_file(url, file_path):
    print(f"Download {url}")

    r = requests.get(url, stream=True)
    with open(file_path, "wb") as f:
        f.write(r.content)

    success = (r.status_code == 200)

    return success


def raise_error(msg):
    instruction_url = "https://github.com/iPERDance/iPERCore/docs/manually_download_datasets.md"
    print(f"{msg} Please manually download all stuffs follow the instruction in {instruction_url}")
    sys.exit(0)
