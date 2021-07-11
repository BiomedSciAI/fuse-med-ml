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

import logging
from typing import List


class FuseProstateXTask():
    tasks = {}
    def __init__(self, task_name: str, version: int):
        self._task_name, self._task_version, self._task_mapping, self._task_class_names = \
            self.get_task(task_name, version)

    def name(self):
        return self._task_name + "_"  + str(self._task_version)

    def class_names(self):
        return self._task_class_names

    def num_classes(self):
        return len(self._task_class_names)

    def mapping(self):
        return self._task_mapping

    @classmethod
    def register(cls, name: str, version: int, mapping: List, class_names: List[str]):
        key = (name, version)
        assert key not in cls.tasks
        cls.tasks[key] = (name, version, mapping, class_names)

    @classmethod
    def get_task(cls, task_name: str, version: int):
        key = (task_name, version)
        if key not in cls.tasks:
            msg = f'Task not found - list of tasks: {list(cls.tasks.keys())}'
            logging.getLogger('Fuse').error(msg)
            raise Exception(msg)

        return cls.tasks[key]



#DO NOT CHANGE TASKS!!!!
GLEASON_SCORE = ['HIGH','LOW','BENIGN']
GLEASON_SCORE_VER_0 = [['HIGH'], ['LOW'],['BENIGN']],
CLINSIG_VER_0 = [['HIGH'], ['LOW']],


FuseProstateXTask.register('gleason_score', 0, GLEASON_SCORE_VER_0, ['HIGH','LOW','BENIGN'])
FuseProstateXTask.register('ClinSig', 0, CLINSIG_VER_0, ['HIGH','LOW'])
if __name__ == '__main__':
    mp_task = FuseProstateXTask('gleason_score', 0)
    print(mp_task.name())
    print(mp_task.class_names())
    print(len(mp_task.class_names()))
    print(mp_task.mapping())