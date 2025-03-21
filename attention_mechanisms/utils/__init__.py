import flax.linen as nn

xavier_uniform = nn.initializers.variance_scaling(
    scale=1.0,
    mode='fan_avg',
    distribution='uniform'
)

__all__ = ["xavier_uniform"]

