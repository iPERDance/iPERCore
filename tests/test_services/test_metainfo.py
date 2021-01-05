# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import unittest

from iPERCore.services.options import meta_info


class TestMetaInfo(unittest.TestCase):

    def test_01_parse_ref_full(self):
        """
        test the full ref_input case.

        Returns:

        """

        ref_input = "path?=/path1,name?=name1,audio?=/audio1,fps?=30,pose_fc?=300,cam_fc?=100|" \
                    "path?=/path2,name?=name2,audio?=/audio2,fps?=25,pose_fc?=200,cam_fc?=50"

        ref_meta_gt = [
            meta_info.RefMetaInputInfo(path="/path1", name="name1", audio="/audio1", fps=30, pose_fc=300, cam_fc=100),
            meta_info.RefMetaInputInfo(path="/path2", name="name2", audio="/audio2", fps=25, pose_fc=200, cam_fc=50)
        ]

        ref_meta_parse = meta_info.parse_ref_input(ref_input)

        for meta_gt, meta_parse in zip(ref_meta_gt, ref_meta_parse):
            # print(meta_gt)
            # print(meta_parse)

            self.assertEqual(meta_gt == meta_parse, True)

    def test_02_parse_ref_only_path(self):
        """
        test the ref_input with only path case.

        Returns:

        """

        ref_input = "/path1|/path2"

        ref_meta_gt = [
            meta_info.RefMetaInputInfo(path="/path1", name="path1"),
            meta_info.RefMetaInputInfo(path="/path2", name="path2")
        ]

        ref_meta_parse = meta_info.parse_ref_input(ref_input)

        for meta_gt, meta_parse in zip(ref_meta_gt, ref_meta_parse):
            # print(meta_gt)
            # print(meta_parse)

            self.assertEqual(meta_gt == meta_parse, True)

    def test_03_parse_ref_only_one_path(self):
        """

            ref_input = "/path1"

        Returns:

        """

        ref_input = "/path1"

        ref_meta_gt = [
            meta_info.RefMetaInputInfo(path="/path1", name="path1")
        ]

        ref_meta_parse = meta_info.parse_ref_input(ref_input)

        for meta_gt, meta_parse in zip(ref_meta_gt, ref_meta_parse):
            # print(meta_gt)
            # print(meta_parse)

            self.assertEqual(meta_gt == meta_parse, True)

    def test_04_parse_ref_missing_key_value(self):
        """

            ref_input = "/path1"

        Returns:

        """

        ref_input = "path?=/path1,audio?=/audio1,pose_fc?=300|" \
                    "path?=/path2,audio?=/audio2,pose_fc?=200"

        ref_meta_gt = [
            meta_info.RefMetaInputInfo(path="/path1", name="path1", audio="/audio1", pose_fc=300),
            meta_info.RefMetaInputInfo(path="/path2", name="path2", audio="/audio2", pose_fc=200)
        ]

        ref_meta_parse = meta_info.parse_ref_input(ref_input)

        for meta_gt, meta_parse in zip(ref_meta_gt, ref_meta_parse):
            # print(meta_gt)
            # print(meta_parse)

            self.assertEqual(meta_gt == meta_parse, True)

    def test_05_parse_ref_missing_value(self):
        """

            ref_input = "/path1"

        Returns:

        """

        ref_input = "path?=/path1,audio?=/audio1,pose_fc?=|" \
                    "path?=/path2,audio?=/audio2,pose_fc?="

        ref_meta_gt = [
            meta_info.RefMetaInputInfo(path="/path1", name="path1", audio="/audio1"),
            meta_info.RefMetaInputInfo(path="/path2", name="path2", audio="/audio2")
        ]

        ref_meta_parse = meta_info.parse_ref_input(ref_input)

        for meta_gt, meta_parse in zip(ref_meta_gt, ref_meta_parse):
            # print(meta_gt)
            # print(meta_parse)

            self.assertEqual(meta_gt == meta_parse, True)

    def test_06_parse_ref_with_warning(self):
        """

            ref_input = "/path1"

        Returns:

        """

        ref_input = "/path1,/audio1,pose_fc?=300|" \
                    "/path2,/audio2,pose_fc?=200"

        ref_meta_gt = [
            meta_info.RefMetaInputInfo(path="/path1", name="path1", pose_fc=300),
            meta_info.RefMetaInputInfo(path="/path2", name="path2", pose_fc=200)
        ]

        ref_meta_parse = meta_info.parse_ref_input(ref_input)

        for meta_gt, meta_parse in zip(ref_meta_gt, ref_meta_parse):
            self.assertEqual(meta_gt == meta_parse, True)

    def test_07_parse_src_full(self):
        src_input = "path?=/path1,bg_path?=/bg_path1|" \
                    "path?=/path2,bg_path?=/bg_path2"

        src_meta_gt = [
            meta_info.SrcMetaInputInfo(path="/path1", bg_path="/bg_path1"),
            meta_info.SrcMetaInputInfo(path="/path2", bg_path="/bg_path2")
        ]

        src_meta_parse = meta_info.parse_src_input(src_input)

        for meta_gt, meta_parse in zip(src_meta_gt, src_meta_parse):
            print(meta_gt)
            print(meta_parse)

            self.assertEqual(meta_gt == meta_parse, True)

    def test_08_parse_src_only_path(self):
        src_input = "/path1|" \
                    "/path2"

        src_meta_gt = [
            meta_info.SrcMetaInputInfo(path="/path1"),
            meta_info.SrcMetaInputInfo(path="/path2")
        ]

        src_meta_parse = meta_info.parse_src_input(src_input)

        for meta_gt, meta_parse in zip(src_meta_gt, src_meta_parse):
            print(meta_gt)
            print(meta_parse)

            self.assertEqual(meta_gt == meta_parse, True)

    def test_09_parse_src_missing_key_value(self):
        src_input = "/path1|" \
                    "/path2"

        src_meta_gt = [
            meta_info.SrcMetaInputInfo(path="/path1"),
            meta_info.SrcMetaInputInfo(path="/path2")
        ]

        src_meta_parse = meta_info.parse_src_input(src_input)

        for meta_gt, meta_parse in zip(src_meta_gt, src_meta_parse):
            print(meta_gt)
            print(meta_parse)

            self.assertEqual(meta_gt == meta_parse, True)

    def test_10_parse_src_missing_value(self):
        src_input = "/path1,bg_path?=|" \
                    "/path2,bg_path?=/bg_path2"

        src_meta_gt = [
            meta_info.SrcMetaInputInfo(path="/path1"),
            meta_info.SrcMetaInputInfo(path="/path2", bg_path="/bg_path2")
        ]

        src_meta_parse = meta_info.parse_src_input(src_input)

        for meta_gt, meta_parse in zip(src_meta_gt, src_meta_parse):
            print(meta_gt)
            print(meta_parse)

            self.assertEqual(meta_gt == meta_parse, True)


if __name__ == '__main__':
    unittest.main()
