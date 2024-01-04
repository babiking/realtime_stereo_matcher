from model.mobile_stereo_net import MobileStereoNet


def build_model(model_config):
    if model_config["type"] == "MobileStereoNet":
        return MobileStereoNet()
    else:
        raise NotImplementedError("unsupport model: {}".format(model_config["type"]))
