"""Public regression for default Janus padding color."""

from transformers.models.janus.image_processing_janus import JanusImageProcessor


def test_background_uses_resolved_image_mean():
    processor = JanusImageProcessor()
    expected = tuple(int(channel * 255) for channel in processor.image_mean)
    assert processor.background_color == expected
