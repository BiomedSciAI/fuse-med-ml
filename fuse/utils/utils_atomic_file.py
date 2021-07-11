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

import gzip
import os
import threading


class FuseUtilsAtomicFileWriter:
    """Writes a file to filename only on successful completion"""

    def __init__(self, filename: str):
        self.filename = filename
        self.temp_filename = f'{filename}_{os.getpid()}_{threading.get_ident() }.tmp'

    def __enter__(self):
        if self.filename.endswith('.gz'):
            self.filehandle = gzip.open(self.temp_filename, 'wb')
        else:
            self.filehandle = open(self.temp_filename, 'wb')
        return self.filehandle

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is None:
            self.filehandle.close()
            os.replace(self.temp_filename, self.filename)
        else:
            try:
                self.filehandle.close()
            finally:
                os.unlink(self.temp_filename)
