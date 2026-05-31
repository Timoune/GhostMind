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

FONT_UI   = ("Segoe UI", 12)
FONT_CHAT = ("Consolas", 11)

WINDOW_WIDTH  = 1000
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
        self.loop    = None

        # Loading animation state
        self._loading_active   = False
        self._loading_frame    = None
        self._loading_label    = None
        self._loading_anim_id  = None
        self._loading_agent    = None   # current multi-agent phase label

        # Stop / cancellation state
        self._current_future  = None
        self._current_prompt  = ""
        self._stop_mode       = None

        # HITL state
        self._approval_frame           = None
        self._current_approval_request = None
        self._hitl_modify_pending      = False   # restore prompt after pipeline returns

        # row 0 = chat, row 1 = stop bar (hidden), row 2 = input bar
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._shutdown)
        self._start_runtime()

    # =====================================================
    # UI BUILD
    # =====================================================

    def _build_ui(self):

        # ââ Chat area ââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        self.chat_frame = ctk.CTkScrollableFrame(
            self,
            fg_color="#151515",
        )
        self.chat_frame.grid(
            row=0, column=0,
            sticky="nsew",
            padx=10, pady=10,
        )

        # ââ Stop bar (row 1, hidden until generation starts) âââââââââââââââ
        self.stop_bar = ctk.CTkFrame(
            self,
            fg_color="#0e0e0e",
            corner_radius=8,
            height=44,
        )
        self.stop_bar.grid_columnconfigure(0, weight=1)

        _stop_inner = ctk.CTkFrame(self.stop_bar, fg_color="transparent")
        _stop_inner.grid(row=0, column=0, sticky="e", padx=10, pady=6)

        self.stop_clear_btn = ctk.CTkButton(
            _stop_inner,
            text="â¹  Stop",
            width=110, height=30,
            fg_color="#2a0f0f",
            hover_color="#4a1a1a",
            text_color="#ff7070",
            font=("Segoe UI", 11, "bold"),
            corner_radius=6,
            command=lambda: self._stop_generation("clear"),
        )
        self.stop_clear_btn.pack(side="left", padx=(0, 8))

        self.stop_refine_btn = ctk.CTkButton(
            _stop_inner,
            text="â  Refine Prompt",
            width=148, height=30,
            fg_color="#0f1e2a",
            hover_color="#1a3044",
            text_color="#8ecfff",
            font=("Segoe UI", 11, "bold"),
            corner_radius=6,
            command=lambda: self._stop_generation("refine"),
        )
        self.stop_refine_btn.pack(side="left")

        # ââ Input bar (row 2) ââââââââââââââââââââââââââââââââââââââââââââââ
        self.bottom_frame = ctk.CTkFrame(
            self,
            fg_color="#111111",
            height=70,
        )
        self.bottom_frame.grid(
            row=2, column=0,
            sticky="ew",
            padx=10, pady=(0, 10),
        )
        self.bottom_frame.grid_columnconfigure(0, weight=1)

        self.input_box = ctk.CTkTextbox(
            self.bottom_frame,
            height=50,
            font=FONT_UI,
        )
        self.input_box.grid(
            row=0, column=0,
            sticky="ew",
            padx=(10, 5), pady=10,
        )
        self.input_box.bind("<Return>", self._handle_enter)

        self.send_button = ctk.CTkButton(
            self.bottom_frame,
            text="Send",
            width=90,
            command=self._send_message,
        )
        self.send_button.grid(
            row=0, column=1,
            padx=(0, 10), pady=10,
        )

    # =====================================================
    # RUNTIME START
    # =====================================================

    def _start_runtime(self):
        threading.Thread(
            target=self._runtime_thread,
            daemon=True,
        ).start()

    def _runtime_thread(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.runtime = Runtime()

        async def start_runtime():
            await self.runtime.initialize()

            # Wire GUI callbacks into runtime subsystems
            self.runtime.set_hitl_callback(self._handle_approval_request)
            self.runtime.set_agent_update_callback(self._handle_agent_update)

            asyncio.create_task(self.runtime.start())

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
        text = self.input_box.get("1.0", "end-1c").strip()
        if not text:
            return

        self._current_prompt  = text
        self._stop_mode       = None
        self._loading_agent   = None
        self._hitl_modify_pending = False

        self.input_box.delete("1.0", "end")
        self.input_box.configure(state="disabled")
        self.send_button.configure(state="disabled")

        self._append_message("USER", text, "#dddddd")
        self._show_loading()
        self._show_stop_bar()

        threading.Thread(
            target=self._process_ai_response,
            args=(text,),
            daemon=True,
        ).start()

    # =====================================================
    # AI PROCESSING
    # =====================================================

    def _process_ai_response(self, text):
        if not self.runtime or not self.loop:
            return

        future = asyncio.run_coroutine_threadsafe(
            self.runtime.think(text),
            self.loop,
        )
        self._current_future = future

        try:
            response = future.result(timeout=300)
        except concurrent.futures.CancelledError:
            self._current_future = None
            return
        except Exception as e:
            response = f"[ERROR]\n{e}"

        self._current_future = None

        def _on_response():
            self._hide_loading()
            self._hide_stop_bar()
            self._hide_approval_dialog()   # clean up if still showing

            if self._hitl_modify_pending:
                # User chose "Modify" â restore prompt instead of showing denial
                self._hitl_modify_pending = False
                self.input_box.configure(state="normal")
                self.input_box.delete("1.0", "end")
                self.input_box.insert("1.0", self._current_prompt)
                self._append_message(
                    "SYSTEM",
                    "Action paused â modify your prompt above to reduce risk, then resend.",
                    "#888888",
                )
            else:
                self._append_message("MINI VON", response, "#8ecfff")
                self.input_box.configure(state="normal")

            self.send_button.configure(state="normal")
            self.input_box.focus()

        self.after(0, _on_response)

    # =====================================================
    # STOP BAR
    # =====================================================

    def _show_stop_bar(self):
        self.stop_bar.grid(
            row=1, column=0,
            sticky="ew",
            padx=10, pady=(0, 4),
        )

    def _hide_stop_bar(self):
        self.stop_bar.grid_remove()

    def _stop_generation(self, mode: str):
        self._stop_mode = mode

        if self._current_future is not None:
            self._current_future.cancel()

        self._hide_loading()
        self._hide_stop_bar()
        self.input_box.configure(state="normal")
        self.send_button.configure(state="normal")

        if mode == "refine":
            self.input_box.delete("1.0", "end")
            self.input_box.insert("1.0", self._current_prompt)
            self._append_message(
                "SYSTEM",
                "Generation stopped â refine your prompt above and resend.",
                "#888888",
            )
        else:
            self._append_message("SYSTEM", "Generation stopped.", "#888888")

        self.input_box.focus()
        self._stop_mode = None

    # =====================================================
    # LOADING ANIMATION
    # =====================================================

    def _show_loading(self):
        self._loading_active = True
        self._loading_agent  = None

        self._loading_frame = ctk.CTkFrame(
            self.chat_frame,
            fg_color="#1a1a1a",
            corner_radius=10,
        )
        self._loading_frame.pack(fill="x", padx=6, pady=6)

        header = ctk.CTkLabel(
            self._loading_frame,
            text="MINI VON",
            font=("Segoe UI", 10, "bold"),
            text_color="#888888",
            anchor="w",
        )
        header.pack(anchor="w", padx=10, pady=(8, 2))

        self._loading_label = ctk.CTkLabel(
            self._loading_frame,
            text="",
            font=("Consolas", 13),
            text_color="#8ecfff",
            anchor="w",
        )
        self._loading_label.pack(fill="x", padx=14, pady=(0, 10))

        self._animate_loading(0)

    def _animate_loading(self, step):
        if not self._loading_active:
            return

        agent = self._loading_agent
        if agent:
            # Multi-agent mode: show agent name + spinner
            dots = ["â ", "â ", "â ¹", "â ¸", "â ¼", "â ´", "â ¦", "â §", "â ", "â "]
            spinner = dots[step % len(dots)]
            label = f"{spinner}  {agent}"
        else:
            frames = ["â¬¤  â  â", "â  â¬¤  â", "â  â  â¬¤", "â  â¬¤  â"]
            label = frames[step % len(frames)]

        if self._loading_label and self._loading_label.winfo_exists():
            self._loading_label.configure(text=label)

        self._loading_anim_id = self.after(
            120 if agent else 350,
            lambda: self._animate_loading(step + 1),
        )

    def _hide_loading(self):
        self._loading_active = False
        self._loading_agent  = None

        if self._loading_anim_id is not None:
            self.after_cancel(self._loading_anim_id)
            self._loading_anim_id = None

        if self._loading_frame is not None:
            self._loading_frame.destroy()
            self._loading_frame = None
            self._loading_label = None

    # =====================================================
    # MULTI-AGENT: agent update callback
    # =====================================================

    def _handle_agent_update(self, agent_name: str):
        """
        Called from the asyncio event-loop thread when a new agent phase starts.
        Schedules a GUI update on the main thread.
        """
        self.after(0, lambda: self._set_loading_agent(agent_name))

    def _set_loading_agent(self, agent_name: str):
        """Update the loading label to show the active agent phase."""
        self._loading_agent = agent_name

    # =====================================================
    # HITL: approval dialog
    # =====================================================

    def _handle_approval_request(self, request):
        """
        Called from the asyncio event-loop thread when the pipeline needs approval.
        Schedules the dialog on the main (tkinter) thread.
        """
        self.after(0, lambda: self._show_approval_dialog(request))

    def _show_approval_dialog(self, request):
        """Render the HITL approval card inside the chat area."""
        self._current_approval_request = request
        self._hide_approval_dialog()   # remove any stale one

        risk_colors = {
            "critical": ("#3a0a0a", "#ff4444"),
            "high":     ("#2a1500", "#ff8800"),
            "medium":   ("#1a1a00", "#ffcc00"),
        }
        bg_color, accent = risk_colors.get(
            request.risk_level, ("#1a1a1a", "#aaaaaa")
        )

        self._approval_frame = ctk.CTkFrame(
            self.chat_frame,
            fg_color=bg_color,
            border_color=accent,
            border_width=2,
            corner_radius=10,
        )
        self._approval_frame.pack(fill="x", padx=6, pady=6)

        # ââ Header ââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        header = ctk.CTkLabel(
            self._approval_frame,
            text=f"â   HUMAN APPROVAL REQUIRED  â  {request.risk_level.upper()} RISK",
            font=("Segoe UI", 11, "bold"),
            text_color=accent,
            anchor="w",
        )
        header.pack(anchor="w", padx=12, pady=(10, 6))

        # ââ Objective âââââââââââââââââââââââââââââââââââââââââââââââââââââ
        obj_text = request.objective
        if len(obj_text) > 140:
            obj_text = obj_text[:137] + "..."
        ctk.CTkLabel(
            self._approval_frame,
            text=f"Objective: {obj_text}",
            font=("Consolas", 10),
            text_color="#dddddd",
            anchor="w",
            wraplength=840,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 4))

        # ââ Warning âââââââââââââââââââââââââââââââââââââââââââââââââââââââ
        if request.warning:
            ctk.CTkLabel(
                self._approval_frame,
                text=request.warning,
                font=("Segoe UI", 10),
                text_color="#ffcc44",
                anchor="w",
                wraplength=840,
                justify="left",
            ).pack(anchor="w", padx=14, pady=(0, 4))

        # ââ Task list âââââââââââââââââââââââââââââââââââââââââââââââââââââ
        if request.task_titles:
            shown    = request.task_titles[:7]
            overflow = len(request.task_titles) - 7
            task_lines = "\n".join(
                f"  {i+1}. {t}" for i, t in enumerate(shown)
            )
            if overflow > 0:
                task_lines += f"\n  â¦ and {overflow} more"
            ctk.CTkLabel(
                self._approval_frame,
                text=f"Planned tasks:\n{task_lines}",
                font=("Consolas", 10),
                text_color="#aaaaaa",
                anchor="w",
                justify="left",
            ).pack(anchor="w", padx=14, pady=(0, 8))

        # ââ Action buttons ââââââââââââââââââââââââââââââââââââââââââââââââ
        btn_frame = ctk.CTkFrame(self._approval_frame, fg_color="transparent")
        btn_frame.pack(anchor="e", padx=12, pady=(0, 12))

        rid = request.request_id

        ctk.CTkButton(
            btn_frame,
            text="â  Approve",
            width=120, height=32,
            fg_color="#0f2a0f",
            hover_color="#1a441a",
            text_color="#66ff88",
            font=("Segoe UI", 11, "bold"),
            corner_radius=6,
            command=lambda: self._hitl_approve(rid),
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_frame,
            text="â  Deny",
            width=100, height=32,
            fg_color="#2a0f0f",
            hover_color="#4a1a1a",
            text_color="#ff7070",
            font=("Segoe UI", 11, "bold"),
            corner_radius=6,
            command=lambda: self._hitl_deny(rid),
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_frame,
            text="â  Modify",
            width=110, height=32,
            fg_color="#0f1e2a",
            hover_color="#1a3044",
            text_color="#8ecfff",
            font=("Segoe UI", 11, "bold"),
            corner_radius=6,
            command=lambda: self._hitl_modify(rid),
        ).pack(side="left")

    def _hide_approval_dialog(self):
        if self._approval_frame is not None:
            try:
                self._approval_frame.destroy()
            except Exception:
                pass
            self._approval_frame = None

    def _hitl_approve(self, request_id: str):
        """User approved â let the pipeline continue."""
        self._hide_approval_dialog()
        if self.loop:
            self.loop.call_soon_threadsafe(
                self.runtime.hitl_gate.resolve,
                request_id, True, "approved", None,
            )

    def _hitl_deny(self, request_id: str):
        """User denied â pipeline will return a denial message normally."""
        self._hide_approval_dialog()
        if self.loop:
            self.loop.call_soon_threadsafe(
                self.runtime.hitl_gate.resolve,
                request_id, False, "denied", None,
            )

    def _hitl_modify(self, request_id: str):
        """
        User wants to edit the prompt.
        Flag for _on_response to restore the input box instead of
        displaying the denial message.
        """
        self._hide_approval_dialog()
        self._hitl_modify_pending = True
        if self.loop:
            self.loop.call_soon_threadsafe(
                self.runtime.hitl_gate.resolve,
                request_id, False, "modify", None,
            )

    # =====================================================
    # CHAT DISPLAY
    # =====================================================

    def _append_message(self, sender: str, message: str, color: str):

        wrapper = ctk.CTkFrame(
            self.chat_frame,
            fg_color="#1a1a1a",
            corner_radius=10,
        )
        wrapper.pack(fill="x", padx=6, pady=6)

        ctk.CTkLabel(
            wrapper,
            text=sender,
            font=("Segoe UI", 10, "bold"),
            text_color="#888888",
            anchor="w",
        ).pack(anchor="w", padx=10, pady=(8, 2))

        ctk.CTkLabel(
            wrapper,
            text=message,
            justify="left",
            wraplength=850,
            anchor="w",
            font=FONT_CHAT,
            text_color=color,
        ).pack(fill="x", padx=10, pady=(0, 10))

    # =====================================================
    # SHUTDOWN
    # =====================================================

    def _shutdown(self):
        if self.runtime and self.loop:
            async def stop_runtime():
                await self.runtime.stop()

            future = asyncio.run_coroutine_threadsafe(
                stop_runtime(), self.loop
            )
            try:
                future.result(timeout=10)
            except Exception:
                pass

        self.destroy()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    app = GhostMindGUI()
    app.mainloop()
