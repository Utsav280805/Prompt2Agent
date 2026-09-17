"""
Standalone helpers for the generated test suite.

These mirror the name-sanitising logic the code generator uses, so the contract tests
can verify that a configuration will actually produce parseable code. They are copied
here rather than imported so the generated project has no dependency on the generator.
"""

import keyword
import re


def sanitize(name, prefix="x"):
    """Turn an arbitrary label into a valid, non-reserved Python identifier."""
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", (name or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = prefix
    if cleaned[0].isdigit():
        cleaned = "{}_{}".format(prefix, cleaned)
    if keyword.iskeyword(cleaned):
        cleaned = cleaned + "_"
    return cleaned


def tool_class_name(name):
    """Build a CamelCase tool class name, e.g. web_scraper -> WebScraperTool."""
    parts = [p for p in re.split(r"[^0-9a-zA-Z]+", name or "") if p]
    if not parts:
        parts = ["custom"]
    camel = "".join(p[:1].upper() + p[1:] for p in parts)
    if camel[0].isdigit():
        camel = "Tool" + camel
    if not camel.endswith("Tool"):
        camel += "Tool"
    return camel
