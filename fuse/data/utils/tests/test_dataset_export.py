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

import unittest

from tempfile import mkstemp
import pandas as pds


from fuse.utils.file_io.file_io import read_dataframe
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.utils.export import ExportDataset
from fuse.data.pipelines.pipeline_default import PipelineDefault


class TestDatasetExport(unittest.TestCase):
    def test_export_to_dataframe(self):
        # datainfo
        data = {"sample_id": ["a", "b", "c", "d", "e"], "values": [7, 4, 9, 2, 4], "not_important": [12] * 5}
        df = pds.DataFrame(data)

        # create simple pipeline
        op = OpReadDataframe(df)
        pipeline = PipelineDefault("test", [(op, {})])

        # create dataset
        dataset = DatasetDefault(data["sample_id"], dynamic_pipeline=pipeline)
        dataset.create()

        df = df.set_index("sample_id")

        # export dataset - only get
        export_df = ExportDataset.export_to_dataframe(dataset, ["values"])
        export_df = export_df.set_index("data.sample_id")
        for sid in data["sample_id"]:
            self.assertEqual(export_df.loc[sid]["values"], df.loc[sid]["values"])

        # export dataset - including save
        _, filename = mkstemp(suffix=".gz")
        _ = ExportDataset.export_to_dataframe(dataset, ["values"], output_filename=filename)
        export_df = read_dataframe(filename)
        export_df = export_df.set_index("data.sample_id")
        for sid in data["sample_id"]:
            self.assertEqual(export_df.loc[sid]["values"], df.loc[sid]["values"])


if __name__ == "__main__":
    unittest.main()
