"""NeuroZip — task-aware neural compression for brain/biomedical signals.

The codec is trained to preserve *decodable semantic content* (what a frozen judge
model can still retrieve) rather than raw signal fidelity.

`import neurozip` is dependency-free. The frozen drop-in embedding layer is exposed
lazily so importing the package never requires PyTorch:

    from neurozip import NeuroZipLayer   # needs the `ml` extra
"""

__version__ = "0.2.0b5"

__all__ = ["NeuroZipLayer", "__version__"]


def __getattr__(name):
    # Lazy so `import neurozip` (CLI, UI) works without torch installed.
    if name == "NeuroZipLayer":
        from .layer import NeuroZipLayer

        return NeuroZipLayer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
