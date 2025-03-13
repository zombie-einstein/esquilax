try:
    import flax  # noqa: F401
except ImportError:
    _has_flax = False
else:
    _has_flax = True


def requires_flax(x):
    """
    Decorator for functionality that requires flax installation
    """
    if not _has_flax:
        raise ImportError(
            "Flax is required for this functionality. "
            "It can be installed with the extra `pip install esquilax[flax]"
        )

    return x
