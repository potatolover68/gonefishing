from diffproc import PipelineConfig
from diffproc.clean import clean_paragraph


def test_plain_paragraph_collapses_whitespace():
    assert clean_paragraph(
        "Hello   world today friend.", "content", PipelineConfig()
    ) == ("Hello world today friend.")


def test_template_is_removed():
    text = "Hello {{foo}} world today friend."
    assert (
        clean_paragraph(text, "content", PipelineConfig())
        == "Hello world today friend."
    )
