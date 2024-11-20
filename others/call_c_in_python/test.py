import ctypes
from typing import TYPE_CHECKING

# https://stackoverflow.com/questions/76249617/how-to-type-hint-ctypes-pointerctypes-c-int
if TYPE_CHECKING:
    IntPointer = ctypes._Pointer[ctypes.c_int]
else:
    IntPointer = ctypes.POINTER(ctypes.c_int)

lib = ctypes.cdll.LoadLibrary("./libdiv.so")
CMPFUNC = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
)


def py_cmp_func(s: IntPointer, r: IntPointer) -> None:
    """
    Callback function to print quotient and remainder.

    Parameters:
    s (POINTER(c_int)): Pointer to the quotient (integer).
    r (POINTER(c_int)): Pointer to the remainder (integer).

    Returns:
    None
    """
    print(f"Quotient is {s[0]} , remainder is {r[0]}")


cmp_func = CMPFUNC(py_cmp_func)
lib.divide(cmp_func, 3, 5)
