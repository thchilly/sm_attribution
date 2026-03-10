"""Suppress HDF5 C-level error-stack dumps that bypass Python's stderr."""


def suppress_hdf5_diagnostics() -> None:
    """Call ``H5Eset_auto2(H5E_DEFAULT, NULL, NULL)`` via *ctypes*.

    HDF5's automatic error stack printing writes directly to C file-descriptor 2,
    bypassing Python's ``sys.stderr``.  This silences the ``HDF5-DIAG`` blocks
    that flood the terminal when concurrent threads probe non-existent files.
    """
    try:
        import ctypes

        for libname in ("libhdf5.so", "libhdf5.dylib"):
            try:
                lib = ctypes.CDLL(libname)
                lib.H5Eset_auto2(0, None, None)
                return
            except OSError:
                continue
    except Exception:
        pass
