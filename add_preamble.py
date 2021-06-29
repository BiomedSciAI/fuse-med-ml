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


import sys, os
import os.path as path
import fileinput

src_extensions = ['.py']
preamble_signature = '(C) Copyright 2021 IBM Corp.'
preamble =   """\"""
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

\"""

"""

def is_preamble_required_file(f):
    if f == '__init__.py':
        return False
    results = [f.endswith(ext) for ext in src_extensions]
    return True in results

def is_header_missing(f):
    with open(f) as reader:
        lines = reader.read().lstrip().splitlines()
        if len(lines) > 1: return not lines[1].startswith(preamble_signature)
        return True

def get_src_files(dirname):
    src_files = []
    for cur, _dirs, files in os.walk(dirname):
        [src_files.append(path.join(cur,f)) for f in files if is_preamble_required_file(f)]

    return [f for f in src_files if is_header_missing(f)]

def add_preamble(files, header):
    for line in fileinput.input(files, inplace=True):
        if fileinput.isfirstline():
            [ print(h) for h in header.splitlines() ]
        print(line, end="")


if __name__ == "__main__":
    """
    Add preamble to .py files except __init__.py. 
    The script scans for source files recursively from <root dir> and list the files that are missing the preamble.
    Then, it prompts for confirmation, and if 'y', the modification is made inplace.

    usage:
      add_preamble.py <root dir>
    """

    if len(sys.argv) < 2:
        print("usage: %s <root dir>" % sys.argv[0])
        exit()

    args = [path.abspath(arg) for arg in sys.argv]
    root_path = path.abspath(args[1])

    files = get_src_files(root_path)

    print("Files with missing preamble:")
    [print("  - %s" % f) for f in files]

    if len(files) > 0:
        confirm = input("proceed ? [y/N] ")
        if confirm is not "y": exit(0)

        add_preamble(files, preamble)
    else:
        print("None")