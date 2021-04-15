# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

class NetworksFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):

        if network_name == "AttLWB-AdaIN":
            from .generators.attlwb_adain_resunet import AttentionLWBGenerator
            network = AttentionLWBGenerator(*args, **kwargs)

        elif network_name == "AttLWB-SPADE":
            from .generators.attlwb_spade_resunet import AttentionLWBGenerator
            network = AttentionLWBGenerator(*args, **kwargs)

        elif network_name == "AttLWB-Front-SPADE":
            from .generators.attlwb_spade_resunet import AttentionLWBFrontGenerator
            network = AttentionLWBFrontGenerator(*args, **kwargs)

        elif network_name == "AddLWB":
            from .generators.lwb_resunet import AddLWBGenerator
            network = AddLWBGenerator(*args, **kwargs)

        elif network_name == "AvgLWB":
            from .generators.lwb_resunet import AvgLWBGenerator
            network = AvgLWBGenerator(*args, **kwargs)

        elif network_name == "SoftGateAddLWB":
            from .generators.lwb_softgate_resunet import SoftGateAddLWBGenerator
            network = SoftGateAddLWBGenerator(*args, **kwargs)

        elif network_name == "SoftGateAvgLWB":
            from .generators.lwb_softgate_resunet import SoftGateAvgLWBGenerator
            network = SoftGateAvgLWBGenerator(*args, **kwargs)

        elif network_name == "InputConcat":
            from .generators.input_concat_resunet import InputConcatGenerator
            network = InputConcatGenerator(*args, **kwargs)

        elif network_name == "TextureWarping":
            from .generators.texture_warping_resunet import TextureWarpingGenerator
            network = TextureWarpingGenerator(*args, **kwargs)

        elif network_name == "multi_scale":
            from .discriminators import MultiScaleDiscriminator
            network = MultiScaleDiscriminator(*args, **kwargs)

        elif network_name == "patch_global":
            from .discriminators import GlobalDiscriminator
            network = GlobalDiscriminator(*args, **kwargs)

        elif network_name == "patch_global_local":
            from .discriminators import GlobalLocalDiscriminator
            network = GlobalLocalDiscriminator(*args, **kwargs)

        elif network_name == "patch_global_body_head":
            from .discriminators import GlobalBodyHeadDiscriminator
            network = GlobalBodyHeadDiscriminator(*args, **kwargs)

        else:
            raise ValueError("Network %s not recognized." % network_name)

        print("Network %s was created" % network_name)

        return network
