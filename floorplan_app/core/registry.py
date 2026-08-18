from __future__ import annotations

from typing import Callable

from floorplan_app.core.parser_base import FloorplanParser


class ParserRegistry:
    """Single place to discover parsers; the Streamlit UI never imports one directly."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], FloorplanParser]] = {}

    def register(self, name: str, factory: Callable[[], FloorplanParser]) -> None:
        if name in self._factories:
            raise ValueError(f'Parser already registered: {name}')
        self._factories[name] = factory

    def names(self) -> list[str]:
        return list(self._factories)

    def create(self, name: str) -> FloorplanParser:
        try:
            return self._factories[name]()
        except KeyError as exc:
            raise KeyError(f'Unknown parser {name!r}. Available: {self.names()}') from exc


registry = ParserRegistry()


def register_parser(name: str):
    def decorator(factory: Callable[[], FloorplanParser]):
        registry.register(name, factory)
        return factory
    return decorator
