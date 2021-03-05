import unittest

from iPERCore.tools.human_mattors.point_render_parser import PointRenderGCAMattor
from iPERCore.tools.background_inpaintors.mmedit_inpaintors import SuperResolutionInpaintors


class TestMMEditInpaintor(unittest.TestCase):

    def test_01_mmedit_inpaintor(self):

        mattor = PointRenderGCAMattor(cfg_or_path="./assets/configs/mattors/point_render+gca.toml")
        inpaintor = SuperResolutionInpaintors(cfg_or_path="./assets/configs/inpaintors/mmedit_inpainting.toml")

        img_path = "./assets/samples/sources/donald_trump_2/00000.PNG"

        segm_mask, trimap = mattor.run_detection(img_path)

        result, dilated_scaled_mask = inpaintor.run_inpainting(img_path, segm_mask)


if __name__ == '__main__':
    unittest.main()
