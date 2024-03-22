from model.mobile_stereo_net import MobileStereoNet
from model.mobile_stereo_net_v2 import MobileStereoNetV2
from model.mobile_stereo_net_v3 import MobileStereoNetV3
from model.mobile_stereo_net_v3_minus import MobileStereoNetV3Minus
from model.mobile_stereo_net_v4 import MobileStereoNetV4
from model.mobile_stereo_net_v5 import MobileStereoNetV5
from model.mobile_stereo_net_v6 import MobileStereoNetV6
from model.mobile_stereo_net_v7 import MobileStereoNetV7
from model.mobile_stereo_net_v8 import MobileStereoNetV8
from model.mobile_stereo_net_v9 import MobileStereoNetV9
from model.opencv_sgbm_module import OpenCVSGBMModule
from others.fast_acv_net.fast_acv_net import Fast_ACVNet
from others.fast_acv_net.fast_acv_net_simple import FastACVNetSimple
from others.mobile_disp_net_c.mobile_disp_net_c import MobileDispNetC
from others.mobile_disp_net_c.fast_disp_net_c import FastDispNetC
from others.mobile_disp_net_c.fast_disp_net_s import FastDispNetS
from others.fast_mad_net.fast_mad_net import FastMADNet


def build_model(model_config):
    if model_config["type"] == "MobileStereoNet":
        return MobileStereoNet()
    elif model_config["type"] == "MobileStereoNetV2":
        return MobileStereoNetV2(**model_config["parameters"])
    elif model_config["type"] == "MobileStereoNetV3":
        return MobileStereoNetV3(**model_config["parameters"])
    elif model_config["type"] == "MobileStereoNetV3Minus":
        return MobileStereoNetV3Minus(**model_config["parameters"])
    elif model_config["type"] == "MobileStereoNetV4":
        return MobileStereoNetV4(**model_config["parameters"])
    elif model_config["type"] == "MobileStereoNetV5":
        return MobileStereoNetV5(**model_config["parameters"])
    elif model_config["type"] == "MobileStereoNetV6":
        return MobileStereoNetV6(**model_config["parameters"])
    elif model_config["type"] == "MobileStereoNetV7":
        return MobileStereoNetV7(**model_config["parameters"])
    elif model_config["type"] == "MobileStereoNetV8":
        return MobileStereoNetV8(**model_config["parameters"])
    elif model_config["type"] == "MobileStereoNetV9":
        return MobileStereoNetV9(**model_config["parameters"])
    elif model_config["type"] == "OpenCVSGBMModule":
        return OpenCVSGBMModule(**model_config["parameters"])
    elif model_config["type"] == "OtherFastACVNet":
        return Fast_ACVNet(**model_config["parameters"])
    elif model_config["type"] == "OtherFastACVNetSimple":
        return FastACVNetSimple(**model_config["parameters"])
    elif model_config["type"] == "MobileDispNetC":
        return MobileDispNetC(**model_config["parameters"])
    elif model_config["type"] == "FastDispNetC":
        return FastDispNetC(**model_config["parameters"])
    elif model_config["type"] == "FastDispNetS":
        return FastDispNetS(**model_config["parameters"])
    elif model_config["type"] == "FastMADNet":
        return FastMADNet(**model_config["parameters"])
    else:
        raise NotImplementedError("unsupport model: {}".format(model_config["type"]))
