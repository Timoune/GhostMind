import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class ServerLauncher:
    """
    Finds the GGUF model in MiniVon/llm/ and launches the llama.cpp server.

    Expected layout:
        MiniVon/
        ├── llm/
        │   └── *.gguf          ← your model file goes here
        └── GhostMind/
            └── GhostMind_v1/
                └── llm/
                    └── server_launcher.py  ← this file

    Default llm_dir is "../../llm" which resolves to MiniVon/llm/ when
    running from inside GhostMind_v1/.

    Usage:
        launcher = ServerLauncher()
        endpoint = await launcher.start()
        # ... run GhostMind ...
        await launcher.stop()

    GPU offloading:
        Set gpu_layers > 0 to offload layers to GPU (requires CUDA/Metal build).
        gpu_layers=-1 offloads all layers.
    """

    def __init__(
        self,
        llm_dir: str = "../../llm",
        port: int = 8080,
        host: str = "127.0.0.1",
        context_length: int = 4096,
        gpu_layers: int = 0,
        boot_wait_seconds: float = 3.0
    ):
        self.llm_dir = Path(llm_dir).resolve()
        self.port = port
        self.host = host
        self.context_length = context_length
        self.gpu_layers = gpu_layers
        self.boot_wait_seconds = boot_wait_seconds
        self._process: Optional[subprocess.Popen] = None

    def find_model(self) -> Path:
        """Find the GGUF file. Picks the largest if multiple exist."""
        models = list(self.llm_dir.glob("*.gguf"))
        if not models:
            raise FileNotFoundError(
                f"No .gguf model found in: {self.llm_dir}\n"
                f"Place your GGUF file there and try again."
            )
        return max(models, key=lambda p: p.stat().st_size)

    def _find_binary(self) -> str:
        """Locate the llama-server binary in PATH."""
        for name in ["llama-server", "llama_server", "server"]:
            path = shutil.which(name)
            if path:
                return path
        raise FileNotFoundError(
            "llama-server not found in PATH.\n"
            "Build llama.cpp and add it to your PATH, or start the "
            "server manually before launching GhostMind."
        )

    async def start(self) -> str:
        """
        Launch llama.cpp server.
        Returns the base endpoint URL (e.g. "http://127.0.0.1:8080").
        """
        model_path = self.find_model()
        binary = self._find_binary()

        cmd = [
            binary,
            "--model",          str(model_path),
            "--host",           self.host,
            "--port",           str(self.port),
            "--ctx-size",       str(self.context_length),
            "--n-gpu-layers",   str(self.gpu_layers),
            "--log-disable",
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Give the server time to load the model before we send requests
        await asyncio.sleep(self.boot_wait_seconds)

        if self._process.poll() is not None:
            _, stderr = self._process.communicate()
            raise RuntimeError(
                f"llama-server failed to start:\n{stderr.decode()}"
            )

        return f"http://{self.host}:{self.port}"

    async def stop(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
