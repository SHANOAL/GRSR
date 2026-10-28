"""Small registry used by training and checkpoint evaluation."""

from typing import Tuple

from .acmnet import ACM_No_Sigmoid
from .config import load_param
from .dnanet import DNANet, Res_CBAM_block, VGG_CBAM_Block


NETWORK_REGISTRY = {
    "acmnet": ACM_No_Sigmoid,
    "dnanet": DNANet,
}


def available_networks() -> Tuple[str, ...]:
    return tuple(sorted(NETWORK_REGISTRY))


def build_network(
    name: str,
    channel_size: str,
    backbone: str,
    deep_supervision: bool,
    num_classes: int = 1,
    input_channels: int = 3,
):
    """Build a registered downstream network.

    DNANet consumes the width/depth presets; ACMNet uses its default configuration.
    """
    key = name.lower()
    if key not in NETWORK_REGISTRY:
        choices = ", ".join(available_networks())
        raise ValueError(f"Unknown downstream network {name!r}; choices: {choices}")

    if key == "acmnet":
        return ACM_No_Sigmoid(in_channels=input_channels, classes=num_classes)

    nb_filter, num_blocks = load_param(channel_size, backbone)
    block = VGG_CBAM_Block if backbone.startswith("vgg") else Res_CBAM_block
    return DNANet(
        num_classes=num_classes,
        input_channels=input_channels,
        block=block,
        num_blocks=num_blocks,
        nb_filter=nb_filter,
        deep_supervision=deep_supervision,
    )
