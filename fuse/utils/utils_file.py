"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

"""
Fuse Files Utils
"""
import errno
import logging
import os
import shutil
from typing import Iterable

from fuse.utils.utils_misc import FuseUtilsMisc


class FuseUtilsFile:
    @staticmethod
    def create_dir(dir_path: str, error_if_exist: bool = False) -> None:
        """
        Create dir
        :param dir_path: path to dir. either relative or full
        :param error_if_exist: if false will ignore already exist error
        :return: None
        """
        if not os.path.isdir(dir_path):
            try:
                os.makedirs(dir_path)
            except OSError as e:
                if e.errno != errno.EEXIST or error_if_exist:
                    raise

    @staticmethod
    def remove_dir_content(dir_path: str, ignore_files: Iterable[str] = tuple(), force_reset: bool = False) -> None:
        """
        Remove the content of dir_path ignoring the files under ignore_files.
        If force_reset is False, prompts the user for approval before the deletion.

        :param dir_path: path to dir. either relative or full
        :param ignore_files: list of files to ignore (don't delete them)
        :param force_reset: when False (default), asks the user for approval when deleting content.
            Else, delete without prompting.
        :return: None
        """
        lgr = logging.getLogger('Fuse')

        # if no content - do nothing
        files = os.listdir(dir_path)
        files = [file for file in files if file not in ignore_files]
        num_files = len(files)
        if num_files > 0:
            # prompt the user for approval
            force_remove = True if force_reset else FuseUtilsMisc.query_yes_no(
                question=f'Folder {dir_path} contains {num_files} files. Delete anyway?')
            # iterate over all the content and delete it
            failure = False
            if force_remove:
                for filename in files:
                    file_path = os.path.join(dir_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        lgr.error('Failed to delete %s. Reason: %s' % (file_path, e))
                        failure = True

            # in case remove wasn't approved or failed to delete some of the files
            if not force_remove or failure:
                msg = f'Folder {dir_path} is already used, please remove it and rerun the program'
                lgr.error(msg)
                raise Exception(msg)

    @staticmethod
    def create_or_reset_dir(dir_path: str, ignore_files: Iterable[str] = tuple(), force_reset: bool = False) -> None:
        """
        Create dir or reset it if already exists
        :param dir_path: path to dir. either relative or full
        :param ignore_files: list of files to ignore (don't delete them)
        :return: None
        """
        FuseUtilsFile.create_dir(dir_path)
        FuseUtilsFile.remove_dir_content(dir_path, ignore_files, force_reset)
