from __future__ import annotations

__all__ = ["main", "RifeVideoInterpolator", "interpolate_video"]


def main() -> None:
    print("Hello from practical-rife-a6i!")


def __getattr__(name: str):
    # Import lazily so `import practical_rife_a6i` does not pull in torch.
    if name in ("RifeVideoInterpolator", "interpolate_video"):
        from practical_rife_a6i import interpolation

        return getattr(interpolation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
