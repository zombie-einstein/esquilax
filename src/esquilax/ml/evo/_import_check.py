try:
    import evosax  # noqa: F401
except ImportError:
    _has_evosax = False
else:
    _has_evosax = True


def requires_evosax(x):
    """
    Decorator for functionality that requires evosax installation
    """
    if not _has_evosax:
        raise ImportError(
            "Evosax is required for this functionality. "
            "It can be installed with the extra `pip install esquilax[evosax]"
        )

    return x
