# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

def build_mattor(name, *args, **kwargs):

    if name == "schp+gca":
        from .schp_parser import SchpMattor

        mattor = SchpMattor(*args, **kwargs)

    elif name == "point_render+gca":
        from .point_render_parser import PointRenderGCAMattor

        mattor = PointRenderGCAMattor(*args, **kwargs)

    else:
        raise ValueError("{} not found, must be {}".format(name, ["schp+gca", "point_render+gca"]))

    return mattor
