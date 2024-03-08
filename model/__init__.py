from model.mobile_stereo_net import MobileStereoNet
from model.mobile_stereo_net_v2 import MobileStereoNetV2
from model.mobile_stereo_net_v3 import MobileStereoNetV3
from model.mobile_stereo_net_v4 import MobileStereoNetV4
from model.mobile_stereo_net_v5 import MobileStereoNetV5
from model.opencv_sgbm_module import OpenCVSGBMModule
from others.fast_acv_net.fast_acv_net import Fast_ACVNet


def build_model(model_config):
    if model_config["type"] == "MobileStereoNet":
        return MobileStereoNet()
    elif model_config["type"] == "MobileStereoNetV2":
        return MobileStereoNetV2(**model_config["parameters"])
    elif model_config["type"] == "MobileStereoNetV3":
        return MobileStereoNetV3(**model_config["parameters"])
    elif model_config["type"] == "MobileStereoNetV4":
        return MobileStereoNetV4(**model_config["parameters"])
    elif model_config["type"] == "MobileStereoNetV5":
        return MobileStereoNetV5(**model_config["parameters"])
    elif model_config["type"] == "OpenCVSGBMModule":
        return OpenCVSGBMModule(**model_config["parameters"])
    elif model_config["type"] == "OtherFastACVNet":
        return Fast_ACVNet(**model_config["parameters"])
    else:
        raise NotImplementedError("unsupport model: {}".format(model_config["type"]))
