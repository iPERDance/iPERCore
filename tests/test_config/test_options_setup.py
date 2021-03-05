import unittest
from easydict import EasyDict
from iPERCore.services.options.options_setup import update_cfg


def compare(dict_a, dict_b):

    if type(dict_a) != type(dict_b):
        return False

    keys_a = sorted(dict_a.keys())
    keys_b = sorted(dict_b.keys())

    if len(keys_a) != len(keys_b):
        return False

    for ka, kb in zip(keys_a, keys_b):
        print(ka, kb)
        if ka != kb:
            return False

        va = dict_a[ka]
        vb = dict_b[kb]

        if isinstance(va, dict):
            if not compare(dict_a[ka], dict_b[kb]):
                return False
        else:
            if va != vb:
                return False

    return True


class TestOptionsSetup(unittest.TestCase):

    def test_01_update_cfg(self):
        cfg = {
            "a": 0,
            "b": 1,
            "c": 2,
            "sub1": {
                "sub21": {
                    "sub31": "sub1->sub21->sub31",
                    "sub32": "sub1->sub22->sub32",
                },
                "sub22": {
                    "sub31": "sub1->sub22->sub31"
                }
            },
        }

        opt = EasyDict({
            "sub1.sub21.sub31": "sub1.sub21.sub31",
            "sub1.sub21.sub32": "sub1.sub22.sub32",
            "sub1.sub23": "sub1.sub23"
        })

        result = {
            "a": 0,
            "b": 1,
            "c": 2,
            "sub1": {
                "sub21": {
                    "sub31": "sub1.sub21.sub31",
                    "sub32": "sub1.sub22.sub32",
                },
                "sub22": {
                    "sub31": "sub1->sub22->sub31"
                },
                "sub23": "sub1.sub23"
            },
        }

        # updated_cfg = update_cfg(opt, cfg)
        update_cfg(opt, cfg)

        print(cfg)

        print(result)

        self.assertEqual(compare(cfg, result), True)


if __name__ == '__main__':
    unittest.main()
