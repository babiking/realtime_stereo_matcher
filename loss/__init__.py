from loss.loss import SequenceLoss


def build_loss_function(loss_config):
    loss_type = loss_config["type"]

    if loss_type == "SequenceLoss":
        return SequenceLoss(**loss_config["parameters"])
    else:
        raise NotImplementedError(f"invalid loss type: {loss_type}!")
