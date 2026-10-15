"""ngspice through its shared library (the sharedspice API), loaded with ctypes.

The library is found through NGSPICE_PATH (a path to ngspice.dll or
libngspice.so), then KiCad's bundled copy on Windows, then the system library
search. ngspice is not reentrant, so every simulation holds one process-wide
lock.
"""

from __future__ import annotations

import ctypes as C
import ctypes.util
import glob
import os
import threading
from pathlib import Path

import numpy as np


class NgspiceError(RuntimeError):
    pass


class _NgComplex(C.Structure):
    _fields_ = [("cx_real", C.c_double), ("cx_imag", C.c_double)]


class _VectorInfo(C.Structure):
    _fields_ = [
        ("v_name", C.c_char_p),
        ("v_type", C.c_int),
        ("v_flags", C.c_short),
        ("v_realdata", C.POINTER(C.c_double)),
        ("v_compdata", C.POINTER(_NgComplex)),
        ("v_length", C.c_int),
    ]


_SendChar = C.CFUNCTYPE(C.c_int, C.c_char_p, C.c_int, C.c_void_p)
_SendStat = C.CFUNCTYPE(C.c_int, C.c_char_p, C.c_int, C.c_void_p)
_ControlledExit = C.CFUNCTYPE(C.c_int, C.c_int, C.c_bool, C.c_bool, C.c_int, C.c_void_p)


def find_library() -> str | None:
    env = os.environ.get("NGSPICE_PATH")
    if env:
        return env
    if os.name == "nt":
        found = sorted(glob.glob(r"C:\Program Files\KiCad\*\bin\ngspice.dll"), key=_kicad_version)
        if found:
            return found[-1]
    return ctypes.util.find_library("ngspice")


def _kicad_version(path: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in Path(path).parents[1].name.split("."))
    except ValueError:
        return (0,)


class Ngspice:
    _instance: Ngspice | None = None
    _lock = threading.Lock()

    def __init__(self, path: str) -> None:
        if os.name == "nt":
            os.add_dll_directory(str(Path(path).parent))
        self._lib = C.CDLL(path)
        self.path = path
        self._output: list[str] = []
        # Keep references to the callbacks, or ctypes frees them while ngspice still calls them.
        self._cb_char = _SendChar(self._on_char)
        self._cb_stat = _SendStat(lambda text, ident, user: 0)
        self._cb_exit = _ControlledExit(self._on_exit)
        lib = self._lib
        lib.ngSpice_Init.argtypes = [_SendChar, _SendStat, _ControlledExit, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
        lib.ngSpice_Init.restype = C.c_int
        lib.ngSpice_Command.argtypes = [C.c_char_p]
        lib.ngSpice_Command.restype = C.c_int
        lib.ngSpice_Circ.argtypes = [C.POINTER(C.c_char_p)]
        lib.ngSpice_Circ.restype = C.c_int
        lib.ngGet_Vec_Info.argtypes = [C.c_char_p]
        lib.ngGet_Vec_Info.restype = C.POINTER(_VectorInfo)
        if lib.ngSpice_Init(self._cb_char, self._cb_stat, self._cb_exit, None, None, None, None) != 0:
            raise NgspiceError("ngSpice_Init failed")

    @classmethod
    def get(cls) -> Ngspice:
        with cls._lock:
            if cls._instance is None:
                path = find_library()
                if path is None:
                    raise NgspiceError("ngspice shared library not found; set NGSPICE_PATH")
                cls._instance = cls(path)
            return cls._instance

    def _on_char(self, text: bytes, ident: int, user: int) -> int:
        self._output.append(text.decode(errors="replace"))
        return 0

    def _on_exit(self, status: int, unload: bool, quit_: bool, ident: int, user: int) -> int:
        self._output.append(f"ngspice requested exit with status {status}")
        return 0

    def version_line(self) -> str:
        return next((line for line in self._output if "ngspice-" in line), "")

    def run(self, netlist: list[str], vectors: list[str]) -> dict[str, np.ndarray]:
        """Load a netlist (title line first, .end last), run its analysis, return copies of the vectors."""
        with self._lock:
            self._output.clear()
            lines = [line.encode() for line in netlist]
            arr = (C.c_char_p * (len(lines) + 1))(*lines, None)
            try:
                if self._lib.ngSpice_Circ(arr) != 0:
                    raise NgspiceError("could not load the circuit:\n" + self._tail())
                if self._lib.ngSpice_Command(b"run") != 0:
                    raise NgspiceError("simulation failed:\n" + self._tail())
                out = {}
                for name in vectors:
                    # The returned struct is reused by the next call, so copy the data now.
                    ptr = self._lib.ngGet_Vec_Info(name.encode())
                    if not ptr:
                        raise NgspiceError(f"no vector {name!r}:\n" + self._tail())
                    info = ptr.contents
                    n = info.v_length
                    if info.v_compdata:
                        data = np.ctypeslib.as_array(C.cast(info.v_compdata, C.POINTER(C.c_double)), shape=(n * 2,))
                        out[name] = data[0::2] + 1j * data[1::2]
                    else:
                        out[name] = np.ctypeslib.as_array(info.v_realdata, shape=(n,)).copy()
                if any("rror" in line for line in self._output):
                    raise NgspiceError("ngspice reported an error:\n" + self._tail())
                return out
            finally:
                self._lib.ngSpice_Command(b"remcirc")
                self._lib.ngSpice_Command(b"destroy all")

    def _tail(self) -> str:
        return "\n".join(self._output[-20:])


def available() -> bool:
    try:
        Ngspice.get()
        return True
    except (NgspiceError, OSError):
        return False


def fmt(value: float) -> str:
    """A SPICE number. Plain exponent notation: SPICE reads 'm' as milli, never mega."""
    return repr(float(value))
