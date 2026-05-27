# =========================================================
# FILE: gui_app.py
# LOCATION: GhostMind/gui_app.py
# =========================================================

import asyncio
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

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_ui()

        self.protocol("WM_DELETE_WINDOW", self._shutdown)

        self._start_runtime()

    # =====================================================
    # UI
    # =====================================================

    def _build_ui(self):

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

        self.bottom_frame = ctk.CTkFrame(
            self,
            fg_color="#111111",
            height=70
        )

        self.bottom_frame.grid(
            row=1,
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

        self.input_box.delete("1.0", "end")

        self._append_message(
            "USER",
            text,
            "#dddddd"
        )

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

        try:
            response = future.result()

        except Exception as e:
            response = f"[ERROR]\n{e}"

        self.after(
            0,
            lambda: self._append_message(
                "MINI VON",
                response,
                "#8ecfff"
            )
        )

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

            asyncio.run_coroutine_threadsafe(
                stop_runtime(),
                self.loop
            )

        self.destroy()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    app = GhostMindGUI()
    app.mainloop()
