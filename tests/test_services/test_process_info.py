import unittest

from iPERCore.services.options.meta_info import MetaInputInfo, MetaProcess
from iPERCore.services.options.process_info import ProcessInfo


class TestProcessInfo(unittest.TestCase):

    def test_01_deserialize(self):
        meta_input = MetaInputInfo(
            path="/p300/tpami/datasets/iPER/images_HD/004/1/2",
            name="004/1/2"
        )

        meta_process = MetaProcess(meta_input, root_primitives_dir="/p300/tpami/datasets/iPER/primitives")
        proc_info = ProcessInfo(meta_process)
        proc_info.deserialize()

        src_infos = proc_info.convert_to_src_info(num_source=8)


if __name__ == '__main__':
    unittest.main()
