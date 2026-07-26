"""Tests for the parts of the vision adapters that don't need an API key.

The network call is stubbed; what's under test is the shared contract — box
denormalization, page geometry, usage accounting, and Markdown assembly.
"""

from __future__ import annotations

from typing import Any

import pytest

from pdfplay.models import Usage
from pdfplay.parsers.vision_base import VisionParser


class FakeVision(VisionParser):
    id = "fake-vision"
    name = "Fake Vision"
    requires = ()
    env_vars = ()

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def call_model(self, png, prompt, opts):
        self.calls += 1
        assert png[:8] == b"\x89PNG\r\n\x1a\n", "adapter must be handed a PNG"
        return self.payload, Usage(input_tokens=1000, output_tokens=200, model="fake", requests=1)


class FakeGeminiStyle(FakeVision):
    id = "fake-gemini"
    bbox_order = "yxyx"


def test_thousand_scale_xyxy_boxes_map_to_points(borderless):
    parser = FakeVision(
        {"blocks": [{"text": "hello", "kind": "title", "bbox": [0, 0, 500, 100]}], "markdown": "# hello"}
    )
    parsed = parser.parse(borderless.path, [1], parser.resolved_options({}))
    page = parsed.pages[0]
    box = page.blocks[0].bbox

    assert abs(box.x0) < 0.01 and abs(box.y0) < 0.01
    assert abs(box.x1 - page.width / 2) < 0.01  # 500/1000 of the width
    assert abs(box.y1 - page.height / 10) < 0.01
    assert page.blocks[0].kind == "title"
    assert parsed.markdown == "# hello"


def test_unit_scale_boxes_are_detected():
    """Models drift between 0-1000 and 0-1; both must land in the same place."""
    parser = FakeVision({"blocks": [], "markdown": ""})
    box = parser._to_points([0.0, 0.0, 0.5, 0.1], 612.0, 792.0)
    assert abs(box.x1 - 306.0) < 0.01
    assert abs(box.y1 - 79.2) < 0.01


def test_gemini_box_order_is_flipped(borderless):
    parser = FakeGeminiStyle(
        {"blocks": [{"text": "row", "kind": "table", "bbox": [100, 0, 200, 1000]}], "markdown": ""}
    )
    parsed = parser.parse(borderless.path, [1], parser.resolved_options({}))
    page, box = parsed.pages[0], parsed.pages[0].blocks[0].bbox
    # [ymin, xmin, ymax, xmax] -> a full-width band 10-20% down the page.
    assert abs(box.x0) < 0.01
    assert abs(box.x1 - page.width) < 0.01
    assert abs(box.y0 - page.height * 0.1) < 0.01
    assert abs(box.y1 - page.height * 0.2) < 0.01


def test_malformed_boxes_are_warned_about_not_fatal(borderless):
    parser = FakeVision(
        {
            "blocks": [
                {"text": "ok", "kind": "text", "bbox": [0, 0, 10, 10]},
                {"text": "bad", "kind": "text", "bbox": "nonsense"},
                {"text": "short", "kind": "text", "bbox": [1, 2]},
            ],
            "markdown": "",
        }
    )
    parsed = parser.parse(borderless.path, [1], parser.resolved_options({}))
    boxes = [b.bbox for b in parsed.pages[0].blocks]
    assert boxes[0] is not None and boxes[1] is None and boxes[2] is None
    assert len(parsed.pages[0].blocks) == 3, "text must survive even without a box"
    assert len(parsed.warnings) == 2


def test_usage_accumulates_across_pages(borderless):
    parser = FakeVision({"blocks": [], "markdown": "page"})
    parsed = parser.parse(borderless.path, None, parser.resolved_options({}))
    assert parser.calls == len(parsed.pages) >= 2
    assert parsed.usage.input_tokens == 1000 * parser.calls
    assert parsed.usage.requests == parser.calls
    assert parsed.markdown.count("page") == parser.calls


def test_json_fence_is_tolerated():
    assert VisionParser.loads('```json\n{"a": 1}\n```') == {"a": 1}
    assert VisionParser.loads('{"a": 2}') == {"a": 2}


def test_cost_estimate_uses_the_price_table():
    from pdfplay.parsers.anthropic_parser import AnthropicVisionParser

    parser = AnthropicVisionParser()
    # 1M input + 1M output on Opus 5 pricing.
    assert parser.estimate_cost("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)
    assert parser.estimate_cost("some-unknown-model", 1000, 1000) is None
