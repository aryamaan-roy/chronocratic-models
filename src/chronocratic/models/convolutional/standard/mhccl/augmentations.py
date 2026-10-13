"""MHCCL augmentation wiring.

Provides ``_default_mhccl_pair()``, which assembles the view pair the reference
implementation uses. The reference vendors TS-TCC's augmentation module verbatim
and computes both views in ``dataloaderq/augmentations.py::DataTransform``::

    aug_1 = scaling(sample, sigma=1.1)
    aug_2 = jitter(permutation(sample, max_segments=8), sigma=0.8)

The constants are TS-TCC's, so this builder is presently identical to
:func:`_default_tstcc_pair`. It is sourced here rather than imported because the
two models' defaults answer to different upstreams and may diverge.

The weak view drives the query encoder and the strong view the momentum encoder,
which is the asymmetry :class:`RolePairProducer` exists to express.

MHCCL operates on tensors of shape ``(batch, time, channels)``, so the
parameters below use ``channel_dim=-1`` and ``time_dim=1``.
"""

from __future__ import annotations

__all__ = ["_default_mhccl_pair"]

from typing import TYPE_CHECKING

from chronocratic.models.augmentation.primitives import (
    ComposeAugmentation,
    Jitter,
    JitterParameters,
    Permutation,
    PermutationParameters,
    Scaling,
    ScalingParameters,
)
from chronocratic.models.augmentation.producers import RolePairProducer

if TYPE_CHECKING:
    from chronocratic.models.augmentation.base import AugmentationProducer, ViewPair


def _default_mhccl_pair() -> AugmentationProducer[ViewPair]:
    """Build the default MHCCL weak/strong augmentation pair.

    Returns a :class:`RolePairProducer` whose first view applies Gaussian
    scaling (weak) and whose second applies segment permutation followed by
    jitter (strong).

    Two divergences from the reference, each marked ``DIVERGENCE`` at the line
    where it happens: the scale factor varies across channels rather than along
    time, and permuted segments keep each channel's own values.

    Returns:
        A producer that returns :class:`ViewPair` instances when
        :meth:`~AugmentationProducer.produce` is called.
    """
    return RolePairProducer(
        # DIVERGENCE: which axis the scale factor varies along. Here one factor
        # is drawn per (sample, channel) and held fixed over time, which is what
        # scaling means everywhere else in this library.
        #
        # The reference draws one per (sample, timestep) instead, and gives
        # every channel the same one. A factor that changes at each timestep is
        # not a rescaling at all — it is multiplicative noise, which the strong
        # view's ``Jitter`` already supplies. Note that both draw per sample;
        # only the second axis differs.
        first=Scaling(ScalingParameters(sigma=1.1, mean=2.0, per_sample=True, channel_dim=-1)),
        second=ComposeAugmentation(
            [
                # DIVERGENCE: all channels are shuffled together, each keeping
                # its own values. The reference assigns ``pat[0, warp]`` into a
                # ``(C, T)`` row, and numpy broadcasts that one row over every
                # channel: channel 0's shuffled values are copied across, and
                # the other channels' data is thrown away. Harmless in TS-TCC,
                # where it comes from, because those datasets have one channel;
                # not harmless on the multivariate ones MHCCL reports.
                Permutation(PermutationParameters(max_segments=8, time_dim=1)),
                Jitter(JitterParameters(sigma=0.8)),
            ]
        ),
    )
