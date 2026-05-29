# =========================================================
# FILE: gui_app.py
# LOCATION: GhostMind/gui_app.py
# =========================================================

import asyncio
import concurrent.futures
import threading
import tkinter as tk
import customtkinter as ctk

from core.runtime import Runtime

# ---------------------------------------------------------
# UI SETTINGS
# ---------------------------------------------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

FONT_UI = ("Segoe UI", 12)
FONT_CHAT = ("Consolas", 11)

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 720


# =========================================================
# APPLICATION
# =========================================================

class GhostMindGUI(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.title("Mini Von")

        self.configure(fg_color="#101010")

        self.runtime = None
        self.loop = None

        # Loading animation state
        self._loading_active = False
        self._loading_frame = None
        self._loading_label = None
        self._loading_anim_id = None

        # Stop / cancellation state
        self._current_future = None   # concurrent.futures.Future for active think()
        self._current_prompt = ""     # original prompt text (restored on Refine)
        self._stop_mode = None        # "clear" | "refine" | None

        # row 0 = chat, row 1 = stop bar (hidden), row 2 = input bar
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_ui()

        self.protocol("WM_DELETE_WINDOW", self._shutdown)

        self._start_runtime()

    # =====================================================
    # UI
    # =====================================================

    def _build_ui(self):

        # ââ Chat area ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        self.chat_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="#151515"
        )

        self.chat_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=10,
            pady=10
        )

        # ââ Stop bar (row 1, hidden until generation starts) ââââââââââââââââââ
        self.stop_bar = ctk.CTkFrame(
            self,
            fg_color="#0e0e0e",
            corner_radius=8,
            height=44
        )
        # Not gridded yet â shown/hidden dynamically

        self.stop_bar.grid_columnconfigure(0, weight=1)

        _stop_inner = ctk.CTkFrame(self.stop_bar, fg_color="transparent")
        _stop_inner.grid(row=0, column=0, sticky="e", padx=10, pady=6)

        self.stop_clear_btn = ctk.CTkButton(
            _stop_inner,
            text="â¹  Stop",
            width=110,
            height=30,
            fg_color="#2a0f0f",
            hover_color="#4a1a1a",
            text_color="#ff7070",
            font=("Segoe UI", 11, "bold"),
            corner_radius=6,
            command=lambda: self._stop_generation("clear")
        )
        self.stop_clear_btn.pack(side="left", padx=(0, 8))

        self.stop_refine_btn = ctk.CTkButton(
            _stop_inner,
            text="â  Refine Prompt",
            width=148,
            height=30,
            fg_color="#0f1e2a",
            hover_color="#1a3044",
            text_color="#8ecfff",
            font=("Segoe UI", 11, "bold"),
            corner_radius=6,
            command=lambda: self._stop_generation("refine")
        )
        self.stop_refine_btn.pack(side="left")

        # ââ Input bar (row 2) ââââââââââââââââââââââââââââââââââââââââââââââââââ
        self.bottom_frame = ctk.CTkFrame(
            self,
            fg_color="#111111",
            height=70
        )

        self.bottom_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=10,
            pady=(0, 10)
        )

        self.bottom_frame.grid_columnconfigure(0, weight=1)

        self.input_box = ctk.CTkTextbox(
            self.bottom_frame,
            height=50,
            font=FONT_UI
        )

        self.input_box.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(10, 5),
            pady=10
        )

        self.input_box.bind("<Return>", self._handle_enter)

        self.send_button = ctk.CTkButton(
            self.bottom_frame,
            text="Send",
            width=90,
            command=self._send_message
        )

        self.send_button.grid(
            row=0,
            column=1,
            padx=(0, 10),
            pady=10
        )

    # =====================================================
    # RUNTIME START
    # =====================================================

    def _start_runtime(self):

        threading.Thread(
            target=self._runtime_thread,
            daemon=True
        ).start()

    def _runtime_thread(self):

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.runtime = Runtime()

        async def start_runtime():

            await self.runtime.initialize()

            asyncio.create_task(
                self.runtime.start()
            )

        self.loop.run_until_complete(start_runtime())
        self.loop.run_forever()

    # =====================================================
    # SEND
    # =====================================================

    def _handle_enter(self, event):

        if event.state & 0x1:
            return

        self._send_message()
        return "break"

    def _send_message(self):

        text = self.input_box.get(
            "1.0",
            "end-1c"
        ).strip()

        if not text:
            return

        self._current_prompt = text
        self._stop_mode = None

        self.input_box.delete("1.0", "end")

        # Disable input controls while waiting for response
        self.input_box.configure(state="disabled")
        self.send_button.configure(state="disabled")

        self._append_message(
            "USER",
            text,
            "#dddddd"
        )

        self._show_loading()
        self._show_stop_bar()

        threading.Thread(
            target=self._process_ai_response,
            args=(text,),
            daemon=True
        ).start()
    # =====================================================
    # AI PROCESSING
    # =====================================================

    def _process_ai_response(self, text):

        if not self.runtime or not self.loop:
            return

        future = asyncio.run_coroutine_threadsafe(
            self.runtime.think(text),
            self.loop
        )

        self._current_future = future

        try:
            response = future.result(timeout=120)

        except concurrent.futures.CancelledError:
            # User hit Stop â let _stop_generation handle the UI
            self._current_future = None
            return

        except Exception as e:
            response = f"[ERROR]\n{e}"

        self._current_future = None

        def _on_response():
            self._hide_loading()
            self._hide_stop_bar()
            self._append_message("MINI VON", response, "#8ecfff")
            self.input_box.configure(state="normal")
            self.send_button.configure(state="normal")
            self.input_box.focus()

        self.after(0, _on_response)

    # =====================================================
    # STOP BAR
    # =====================================================

    def _show_stop_bar(self):
        """Slide the stop bar in above the input."""
        self.stop_bar.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=10,
            pady=(0, 4)
        )

    def _hide_stop_bar(self):
        """Remove the stop bar from the layout."""
        self.stop_bar.grid_remove()

    def _stop_generation(self, mode: str):
        """
        Cancel the in-flight LLM future and handle the two stop modes:
          - 'clear'  : discard, let the user start fresh
          - 'refine' : restore the original prompt so it can be edited
        """
        self._stop_mode = mode

        if self._current_future is not None:
            self._current_future.cancel()

        # _process_ai_response will return early on CancelledError;
        # we drive all UI changes here on the main thread.
        self._hide_loading()
        self._hide_stop_bar()
        self.input_box.configure(state="normal")
        self.send_button.configure(state="normal")

        if mode == "refine":
            # Restore the original prompt and let the user amend it
            self.input_box.delete("1.0", "end")
            self.input_box.insert("1.0", self._current_prompt)
            self._append_message(
                "SYSTEM",
                "Generation stopped â refine your prompt above and resend.",
                "#888888"
            )
        else:
            # Clear stop â just acknowledge and move on
            self._append_message(
                "SYSTEM",
                "Generation stopped.",
                "#888888"
            )

        self.input_box.focus()
        self._stop_mode = None

    # =====================================================
    # LOADING ANIMATION
    # =====================================================

    def _show_loading(self):
        """Show an animated thinking bubble while the LLM is generating."""

        self._loading_active = True

        self._loading_frame = ctk.CTkFrame(
            self.chat_frame,
            fg_color="#1a1a1a",
            corner_radius=10
        )

        self._loading_frame.pack(
            fill="x",
            padx=6,
            pady=6
        )

        header = ctk.CTkLabel(
            self._loading_frame,
            text="MINI VON",
            font=("Segoe UI", 10, "bold"),
            text_color="#888888",
            anchor="w"
        )

        header.pack(
            anchor="w",
            padx=10,
            pady=(8, 2)
        )

        self._loading_label = ctk.CTkLabel(
            self._loading_frame,
            text="",
            font=("Consolas", 14),
            text_color="#8ecfff",
            anchor="w"
        )

        self._loading_label.pack(
            fill="x",
            padx=14,
            pady=(0, 10)
        )

        self._animate_loading(0)

    def _animate_loading(self, step):
        """Cycle through dot animation frames."""

        if not self._loading_active:
            return

        frames = [
            "â¬¤  â  â",
            "â  â¬¤  â",
            "â  â  â¬¤",
            "â  â¬¤  â",
        ]

        if self._loading_label and self._loading_label.winfo_exists():
            self._loading_label.configure(text=frames[step % len(frames)])

        self._loading_anim_id = self.after(
            350,
            lambda: self._animate_loading(step + 1)
        )

    def _hide_loading(self):
        """Remove the loading bubble and cancel the animation."""

        self._loading_active = False

        if self._loading_anim_id is not None:
            self.after_cancel(self._loading_anim_id)
            self._loading_anim_id = None

        if self._loading_frame is not None:
            self._loading_frame.destroy()
            self._loading_frame = None
            self._loading_label = None

    # =====================================================
    # CHAT DISPLAY
    # =====================================================

    def _append_message(self, sender, message, color):

        wrapper = ctk.CTkFrame(
            self.chat_frame,
            fg_color="#1a1a1a",
            corner_radius=10
        )

        wrapper.pack(
            fill="x",
            padx=6,
            pady=6
        )

        header = ctk.CTkLabel(
            wrapper,
            text=sender,
            font=("Segoe UI", 10, "bold"),
            text_color="#888888",
            anchor="w"
        )

        header.pack(
            anchor="w",
            padx=10,
            pady=(8, 2)
        )

        body = ctk.CTkLabel(
            wrapper,
            text=message,
            justify="left",
            wraplength=850,
            anchor="w",
            font=FONT_CHAT,
            text_color=color
        )

        body.pack(
            fill="x",
            padx=10,
            pady=(0, 10)
        )

    # =====================================================
    # SHUTDOWN
    # =====================================================

    def _shutdown(self):

        if self.runtime and self.loop:

            async def stop_runtime():
                await self.runtime.stop()

            # Block until the runtime has shut down cleanly so aiohttp
            # sessions are closed before the window is destroyed.
            future = asyncio.run_coroutine_threadsafe(
                stop_runtime(),
                self.loop
            )
            try:
                future.result(timeout=10)
            except Exception:
                pass  # best-effort shutdown

        self.destroy()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    app = GhostMindGUI()
    app.mainloop()
