# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

def build_background_inpaintors(name, *args, **kwargs):

    if name == "mmedit_inpainting":
        from .mmedit_inpaintors import SuperResolutionInpaintors

        inpaintors = SuperResolutionInpaintors(*args, **kwargs)

    else:
        raise ValueError(f"{name} is not valid background inpaintors.")

    return inpaintors
