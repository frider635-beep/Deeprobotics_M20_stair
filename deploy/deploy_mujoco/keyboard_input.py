"""Focused X11 window key state, independent of MuJoCo's press-only callback."""
from __future__ import annotations

import ctypes
import os
import sys
import threading


MOTION_KEYS = "824679"
# GLFW reports physical keypad keys independently of the Num Lock state.
KEYPAD_MOTION = {320 + int(key): key for key in MOTION_KEYS}
_VIEWER_LAUNCH_LOCK = threading.Lock()


def install_window_key_filter(keyboard):
    """Reserve teleop keys before MuJoCo's native UI shortcut handler runs.

    Call on the viewer thread, after its GLFW context has been created. The
    passive viewer's Python key_callback runs AFTER the native shortcuts, so
    returning from that callback cannot consume a key. Keep this callback alive
    until the viewer thread exits.
    """
    import glfw

    window = glfw.get_current_context()
    if not window:
        raise RuntimeError("Teleop key filtering needs the viewer's GLFW context")
    reserved = {ord(key.upper()) for key in "zcxrh"}
    reserved.update(KEYPAD_MOTION)
    reserved.update((glfw.KEY_ESCAPE, glfw.KEY_BACKSPACE))
    modifiers = glfw.MOD_CONTROL | glfw.MOD_ALT | glfw.MOD_SUPER
    native_callback = None

    def dispatch(window, key, scancode, action, mods):
        if key in reserved and not mods & modifiers:
            keyboard.on_window_key(key, action)
            return
        if native_callback:
            native_callback(window, key, scancode, action, mods)

    callback = glfw._GLFWkeyfun(dispatch)
    # The public wrapper only returns callbacks installed through Python; the
    # existing MuJoCo callback was installed in C++. Retain the native pointer.
    native_callback = glfw._glfw.glfwSetKeyCallback(window, callback)
    return callback


def launch_teleop_viewer(model, data, keyboard):
    """Launch the standard passive viewer with teleop shortcuts reserved."""
    from mujoco import viewer

    # Install on the creating/rendering thread before any key event is polled.
    # MuJoCo exposes no public pre-shortcut hook. This narrow constructor hook
    # retains its normal renderer, mouse controls, menus and shutdown path.
    with _VIEWER_LAUNCH_LOCK:
        original_simulate = viewer._Simulate
        callbacks = []

        class TeleopSimulate(original_simulate):
            def __init__(self, *args):
                super().__init__(*args)
                callbacks.append(install_window_key_filter(keyboard))

        viewer._Simulate = TeleopSimulate
        try:
            handle = viewer.launch_passive(model, data)
        finally:
            viewer._Simulate = original_simulate
        handle._teleop_key_callbacks = callbacks
        return handle


class X11WindowKeys:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._x11 = ctypes.CDLL("libX11.so.6")
        x = self._x11
        x.XInitThreads()
        x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x.XOpenDisplay.restype = ctypes.c_void_p
        x.XGetInputFocus.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int)]
        x.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        x.XStringToKeysym.argtypes = [ctypes.c_char_p]
        x.XStringToKeysym.restype = ctypes.c_ulong
        x.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x.XKeysymToKeycode.restype = ctypes.c_uint
        x.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._display = x.XOpenDisplay(None)
        if not self._display:
            raise OSError("Cannot connect to X11 display")
        self._window = None
        self._codes = {
            key: x.XKeysymToKeycode(self._display, x.XStringToKeysym(("KP_" + key).encode()))
            for key in MOTION_KEYS
        }

    @classmethod
    def open(cls) -> X11WindowKeys | None:
        if not sys.platform.startswith("linux") or not os.environ.get("DISPLAY"):
            return None
        if os.environ.get("XDG_SESSION_TYPE") == "wayland":
            return None  # XWayland cannot reliably query keys outside its surface.
        try:
            return cls()
        except OSError:
            return None

    def _focus(self) -> int:
        window, revert = ctypes.c_ulong(), ctypes.c_int()
        self._x11.XGetInputFocus(self._display, ctypes.byref(window), ctypes.byref(revert))
        return window.value

    def note_viewer_event(self) -> None:
        # Called exclusively by the MuJoCo window callback, never terminal input.
        with self._lock:
            if self._display:
                focus = self._focus()
                self._window = focus if focus > 1 else None

    def pressed(self) -> set[str]:
        with self._lock:
            if not self._display or self._window is None or self._focus() != self._window:
                return set()
            bits = ctypes.create_string_buffer(32)
            self._x11.XQueryKeymap(self._display, bits)
            if self._focus() != self._window:
                return set()
            return {
                key for key, code in self._codes.items()
                if code and bits.raw[code // 8] & (1 << (code % 8))
            }

    def close(self) -> None:
        with self._lock:
            if self._display:
                self._x11.XCloseDisplay(self._display)
                self._display = None
                self._window = None
