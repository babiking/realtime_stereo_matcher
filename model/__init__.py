from model.mobile_stereo_net import MobileStereoNet
from model.mobile_stereo_net_v2 import MobileStereoNetV2
from model.mobile_stereo_net_v3 import MobileStereoNetV3


def build_model(model_config):
    if model_config["type"] == "MobileStereoNet":
        return MobileStereoNet()
    elif model_config["type"] == "MobileStereoNetV2":
        return MobileStereoNetV2(**model_config["parameters"])
    elif model_config["type"] == "MobileStereoNetV3":
        return MobileStereoNetV3(**model_config["parameters"])
    else:
        raise NotImplementedError("unsupport model: {}".format(model_config["type"]))
