from model.mobile_stereo_net import MobileStereoNet
from model.mobile_raft_net import MobileRaftNet


def build_model(model_config):
    if model_config["type"] == "MobileStereoNet":
        return MobileStereoNet()
    elif model_config["type"] == "MobileRaftNet":
        return MobileRaftNet(**model_config["parameters"])
    else:
        raise NotImplementedError("unsupport model: {}".format(model_config["type"]))
