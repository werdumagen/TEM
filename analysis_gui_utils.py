from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Optional

if not hasattr(tk, "Notebook") and hasattr(ttk, "Notebook"):
    tk.Notebook = ttk.Notebook

class AnalysisConfirmationDialog(tk.Toplevel):
    """A small, borderless dialog with accept/reject buttons."""
    def __init__(self, parent, accept_callback, reject_callback, x, y):
        super().__init__(parent)
        self.accept_callback = accept_callback
        self.reject_callback = reject_callback

        # Make window borderless and stay on top
        self.overrideredirect(True)
        self.wm_attributes("-topmost", True)

        # Positioning
        self.geometry(f"+{int(x)}+{int(y)}")

        # Frame for content
        frame = tk.Frame(self, background='white', highlightbackground="black", highlightthickness=1)
        frame.pack()

        try:
            # Attempt to load icons if available
            # Green checkmark icon
            check_icon_data = b"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAADnSURBVDhPnY5BCsJgEETfZsE3iCDeQDxD8SbeoZ15gJcQvEknl3Zl7wYKSJb58nwzBpkfOElYmJqgmz8hJ04m6A6STpyUaA7STpySaA/STpyEaA/STpyBaA/STpyCaA/STpyBaA/STpyBaA/STpwoBeoGg2/iV0WwM9vxvUvAtb2P6b0LoTVQ5A0oVbQ6/hbNsgV/ycb/1k/gC6jX8f79wJ24+2WqlwCWv5u0FhROO2LgZ1xU8f07gW0e5Z0E0s+B3A/yQ+wlgQ/y1A/q5f/gB/829kFh2BfTQAAAABJRU5ErkJggg=="
            self.check_img = tk.PhotoImage(data=check_icon_data)
            btn_accept = tk.Button(frame, image=self.check_img, command=self.accept_callback, borderwidth=0, relief="flat", bg="white")

            # Red cross icon
            cross_icon_data = b"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAACYSURBVDhPzY1BCsAwCEMv9xVeQPEMxTcUvEBv15kXkLwJ7+7sSxWE+C8L/5nF5H8gA8fS4qQfMBNf9CdoFvGkbyAaxJOnIFrEk6cgWsSTpyAaxJOvIFrEk68gWsSTr6BaxJOvIFrEk68gWsSTr6AaxJMnA7hX5oAZuDmz+W8Crf130/oWQLuAZ38DToPj/5v/6/4BN8V2yEaG41QAAAAASUVORK5CYII="
            self.cross_img = tk.PhotoImage(data=cross_icon_data)
            btn_reject = tk.Button(frame, image=self.cross_img, command=self.reject_callback, borderwidth=0, relief="flat", bg="white")

        except tk.TclError: # Fallback to text if icons fail
            btn_accept = tk.Button(frame, text="✔", command=self.accept_callback, fg="green", relief="flat", bg="white")
            btn_reject = tk.Button(frame, text="✖", command=self.reject_callback, fg="red", relief="flat", bg="white")

        btn_accept.pack(side="left", padx=2, pady=2)
        btn_reject.pack(side="left", padx=2, pady=2)

class _Tooltip:
    """A generic tooltip class for tkinter widgets."""
    def __init__(self, widget: tk.Widget, text: str, *, delay: int = 400):
        self.widget = widget
        self.text = text
        self.delay = max(0, int(delay))
        self._after_id: Optional[str] = None
        self._tip_window: Optional[tk.Toplevel] = None
        self._last_pointer: Optional[tuple[int, int]] = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<Motion>", self._on_motion, add="+")

    def _on_enter(self, event):
        self._last_pointer = (event.x_root, event.y_root)
        self._schedule()

    def _on_leave(self, _event):
        self._cancel()
        self._hide()

    def _on_motion(self, event):
        self._last_pointer = (event.x_root, event.y_root)
        self._position()

    def _schedule(self):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tip_window is not None or not self.text:
            return
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_attributes("-topmost", True)
        label = tk.Label(
            tip, text=self.text, justify="left", background="#ffffe0",
            relief="solid", borderwidth=1, wraplength=360,
        )
        label.pack(ipadx=8, ipady=4)
        self._tip_window = tip
        self._position()

    def _hide(self):
        if self._tip_window is not None:
            self._tip_window.destroy()
            self._tip_window = None

    def _position(self):
        if self._tip_window is None:
            return
        tip = self._tip_window
        tip.update_idletasks()
        width = tip.winfo_reqwidth()
        height = tip.winfo_reqheight()
        if self._last_pointer is not None:
            x, y = self._last_pointer
        else:
            x = self.widget.winfo_rootx() + self.widget.winfo_width()
            y = self.widget.winfo_rooty() + self.widget.winfo_height()
        x += 12
        y += 10
        root = self.widget.winfo_toplevel()
        root.update_idletasks()
        left = root.winfo_rootx()
        top = root.winfo_rooty()
        right = left + root.winfo_width()
        bottom = top + root.winfo_height()
        if x + width > right - 4:
            x = right - width - 4
        if y + height > bottom - 4:
            y = bottom - height - 4
        x = max(x, left + 4)
        y = max(y, top + 4)
        tip.wm_geometry(f"+{int(x)}+{int(y)}")

class HoverTooltip:
    """A dark-style tooltip class."""
    def __init__(self, widget: tk.Widget, text: str, delay: int = 400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id: Optional[str] = None
        self._window: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self._window is not None:
            return
        x = self.widget.winfo_pointerx() + 16
        y = self.widget.winfo_pointery() + 12
        self._window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#2f2f2f", foreground="white",
                         relief="solid", borderwidth=1, padx=6, pady=3, justify=tk.LEFT)
        label.pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None