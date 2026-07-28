import numpy as np

import pytest

from scripts.dover_technical_runner import normalize_checkpoint_state, reshape_technical_clips


def test_technical_clip_layout_transposes_clip_before_channel() -> None:
    technical = np.arange(3 * 6).reshape(3, 6, 1, 1)
    mean = np.zeros((3, 1, 1, 1))
    std = np.ones((3, 1, 1, 1))

    clips = reshape_technical_clips(
        technical,
        mean,
        std,
        num_clips=3,
        clip_len=2,
    )

    assert clips.shape == (3, 3, 2, 1, 1)
    assert clips[0, 1, :, 0, 0].tolist() == [6, 7]
    assert clips[1, 0, :, 0, 0].tolist() == [2, 3]


def test_checkpoint_requires_every_technical_parameter() -> None:
    class Model:
        @staticmethod
        def state_dict():
            return {"technical_backbone.weight": object(), "technical_head.weight": object()}

    normalized = normalize_checkpoint_state(
        Model(),
        {
            "state_dict": {
                "module.technical_backbone.weight": 1,
                "module.technical_head.weight": 2,
                "module.aesthetic_head.weight": 3,
            }
        },
    )
    assert normalized["technical_head.weight"] == 2

    with pytest.raises(RuntimeError, match="checkpoint_missing_technical_weights"):
        normalize_checkpoint_state(Model(), {"technical_backbone.weight": 1})
