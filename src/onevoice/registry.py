"""Maps a config string to a component class.

Adding a model touches no framework code: the wrapper decorates itself with
@register and a YAML config names it.
"""

from __future__ import annotations

from collections.abc import Callable

KINDS = ("vad", "asr", "mt", "tts")

_REGISTRY: dict[str, dict[str, type]] = {kind: {} for kind in KINDS}


def register(kind: str, name: str) -> Callable[[type], type]:
    """Register a component class under (kind, name).

    D17: registering the same pair twice raises. A silent overwrite would mean
    whichever module imported last wins, making the benchmarked model depend on
    import order.
    """
    if kind not in _REGISTRY:
        raise KeyError(f"unknown component kind {kind!r}; expected one of {KINDS}")

    def decorator(cls: type) -> type:
        if name in _REGISTRY[kind]:
            existing = _REGISTRY[kind][name].__name__
            raise ValueError(
                f"{kind} component {name!r} is already registered to {existing}; "
                "pick a different name rather than shadowing it"
            )
        _REGISTRY[kind][name] = cls
        return cls

    return decorator


def resolve(kind: str, name: str) -> type:
    """Return the class registered under (kind, name)."""
    if kind not in _REGISTRY:
        raise KeyError(f"unknown component kind {kind!r}; expected one of {KINDS}")
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        available = ", ".join(registered_names(kind)) or "(none registered)"
        raise KeyError(
            f"no {kind} component named {name!r}; registered: {available}"
        ) from None


def registered_names(kind: str) -> list[str]:
    """Sorted names registered for a kind."""
    if kind not in _REGISTRY:
        raise KeyError(f"unknown component kind {kind!r}; expected one of {KINDS}")
    return sorted(_REGISTRY[kind])


def clear_registry() -> None:
    """Empty the registry. For test isolation only."""
    for kind in _REGISTRY:
        _REGISTRY[kind].clear()


def snapshot() -> dict[str, dict[str, type]]:
    """Copy the current registry state. For test isolation only.

    Pair with restore() so a test that needs a clean registry can put back
    what was there before it, rather than clearing it for every caller that
    runs afterwards in the same process.
    """
    return {kind: dict(names) for kind, names in _REGISTRY.items()}


def restore(state: dict[str, dict[str, type]]) -> None:
    """Replace the registry contents with a previously taken snapshot."""
    for kind in _REGISTRY:
        _REGISTRY[kind].clear()
        _REGISTRY[kind].update(state.get(kind, {}))
