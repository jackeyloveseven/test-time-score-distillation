"""Prompt text helpers for positive and content-only teacher branches."""

from __future__ import annotations

from collections.abc import Mapping


def _content_from_prompt_id(prompt_id: str, prompt_lookup: Mapping[str, str]) -> str:
    return str(prompt_lookup.get(prompt_id, prompt_id))


def parse_prompt_text(
    *,
    prompt: str | None,
    prompt_id: str,
    prompt_template: str,
    prompt_lookup: Mapping[str, str],
) -> str:
    if prompt:
        return prompt
    content = _content_from_prompt_id(prompt_id, prompt_lookup)
    return prompt_template.format(content=content)


def parse_content_prompt_text(
    *,
    content_prompt: str | None,
    prompt: str | None,
    prompt_id: str,
    content_prompt_template: str,
    prompt_lookup: Mapping[str, str],
) -> str:
    if content_prompt:
        return content_prompt
    if prompt:
        raise ValueError(
            "--without-prompt-mode=content_only with a custom --prompt requires "
            "an explicit --content-prompt so the negative branch removes reference/style wording."
        )
    content = _content_from_prompt_id(prompt_id, prompt_lookup)
    return content_prompt_template.format(content=content)


__all__ = ["parse_content_prompt_text", "parse_prompt_text"]
