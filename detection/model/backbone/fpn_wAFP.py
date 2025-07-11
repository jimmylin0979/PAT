from detectron2.modeling.backbone import FPN
import torch.nn.functional as F
from .registry import register_method

_target_class = FPN


@register_method
def forward(self, x, afp_feedback=None):
    """
    Args:
        input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
            feature map tensor for each feature level in high to low resolution order.

    Returns:
        dict[str->Tensor]:
            mapping from feature map name to FPN feature map tensor
            in high to low resolution order. Returned feature names follow the FPN
            paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
            ["p2", "p3", ..., "p6"].
    """
    if afp_feedback is not None:
        bottom_up_features = self.bottom_up(x, afp_feedback)
    else:
        bottom_up_features = self.bottom_up(x)
    results = []
    prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
    results.append(self.output_convs[0](prev_features))

    # Reverse feature maps into top-down order (from low to high resolution)
    for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs)):
        # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
        # Therefore we loop over all modules but skip the first one
        if idx > 0:
            features = self.in_features[-idx - 1]
            features = bottom_up_features[features]
            top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

    if self.top_block is not None:
        if self.top_block.in_feature in bottom_up_features:
            top_block_in_feature = bottom_up_features[self.top_block.in_feature]
        else:
            top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
        results.extend(self.top_block(top_block_in_feature))
    assert len(self._out_features) == len(results)
    return {f: res for f, res in zip(self._out_features, results)}
