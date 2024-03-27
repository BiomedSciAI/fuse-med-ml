import os
from os.path import join, realpath, dirname
import shutil
from typing import List

# from multiprocessing import Lock
from filelock import FileLock
from glob import glob
import psutil

SHARED_MEM_FILES_PREFIX = "OUR_SHARED_MEM_@"
SHARED_MEM_ACTIVE_ENV_VAR = "ACTIVATE_OUR_SHARED_MEM"
SHM_BASE_DIR = "/dev/shm/"

G_lock = FileLock(join(SHM_BASE_DIR, "our_shared_mem_file_lock"))  # Lock()


def get_shared_mem_file_path(file_path: str) -> str:
    """
    copies the file (if needed) to /dev/shm/[filename] which effectively make linux os to load it into RAM
    every following access to the file will actually be on RAM which is much faster

    note - it's up to us to clean up the files, otherwise they will continue to consume RAM until a reboot!

    a nice guide about this topic: https://datawookie.dev/blog/2021/11/shared-memory-docker/

    """

    if SHARED_MEM_ACTIVE_ENV_VAR not in os.environ:
        return file_path

    if os.environ[SHARED_MEM_ACTIVE_ENV_VAR].lower() not in ["1", "t", "true"]:
        return file_path

    with G_lock:
        assert os.path.isfile(
            file_path
        ), f"file_path does not point to a file: file_path={file_path}"

        src_file_size_bytes = os.stat(file_path).st_size
        dest = join(SHM_BASE_DIR, SHARED_MEM_FILES_PREFIX + realpath(file_path))

        if os.path.isfile(dest):
            # it already exists, let's see if the size matches
            dest_file_size_bytes = os.stat(dest).st_size
            if dest_file_size_bytes == src_file_size_bytes:
                return dest
            # we found it, but size does not match, so we need to delete it first, and then copy
            print(
                f"get_shared_mem_location:size mismatch (src bytes {src_file_size_bytes} , dest bytes {dest_file_size_bytes}) - deleting dest"
            )
            os.remove(dest)  # remove this file

        available_memory = psutil.virtual_memory().available
        if available_memory < src_file_size_bytes:
            raise Exception(
                f"get_shared_mem_file_path:requested file size {src_file_size_bytes} bytes is bigger than available RAM {available_memory}"
            )

        print(f"get_shared_mem_location:copying {file_path} to {dest}")
        os.makedirs(dirname(dest), exist_ok=True)
        shutil.copyfile(
            file_path, dest
        )  # note - this fails without exception if the directory at destination isn't found
        return dest


def get_all_shared_mem_files() -> List[str]:
    found = glob(f"{SHM_BASE_DIR}/{SHARED_MEM_FILES_PREFIX}/**/*", recursive=True)
    found = [x for x in found if os.path.isfile(x)]
    return found


def get_shared_memory_info_for_our_files(verbose: bool = True) -> int:
    found = get_all_shared_mem_files()
    print(f"found {len(found)} shared memory files:")
    if 0 == len(found):
        return
    total_bytes = 0
    for fn in found:
        curr_size = os.stat(fn).st_size
        if verbose:
            print(f"{curr_size} bytes, {fn}")
        total_bytes += curr_size

    if verbose:
        print(f"total size found {curr_size/(1024**2):.2f} Mbs")
    return total_bytes


def delete_all_of_our_shared_memory_files() -> None:
    found = get_all_shared_mem_files()
    print(f"found {len(found)} shared memory files:")
    if 0 == len(found):
        return
    for fn in found:
        print(f"deleting {fn}")
        os.remove(fn)


if __name__ == "__main__":
    get_shared_memory_info_for_our_files()
