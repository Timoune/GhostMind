from pathlib import Path
import yaml


class ConfigLoader:
    """
    Loads all YAML configs from the configs/ directory.

    Supports both .yaml and .yaml.md (Obsidian) file formats.

    Key derivation:
        memory.yaml.md  →  key "memory"
        runtime.yaml.md →  key "runtime"

    Root-key unwrapping:
        If a YAML file has a single root key that matches the filename key,
        the root is unwrapped so access paths stay clean.

        Example:
            memory.yaml.md contains { memory: { api: { host: ... } } }
            → stored as configs["memory"] = { api: { host: ... } }
            → accessed as config_loader.get("memory.api.host")  ✓
    """

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs: dict = {}

    def load_all(self):
        seen: set = set()

        for pattern in ("*.yaml.md", "*.yaml"):
            for file in sorted(self.config_dir.glob(pattern)):
                key = file.name.split(".")[0]   # first segment only
                if key in seen:
                    continue
                seen.add(key)
                with open(file, "r", encoding="utf-8") as f:
                    parsed = yaml.safe_load(f)

                # Unwrap single-root-key YAMLs whose root matches the filename
                if (
                    isinstance(parsed, dict)
                    and list(parsed.keys()) == [key]
                ):
                    self.configs[key] = parsed[key]
                else:
                    self.configs[key] = parsed

        self.validate()

    def validate(self):
        required = ["runtime", "cognition", "models", "safety"]
        for name in required:
            if name not in self.configs:
                raise RuntimeError(f"Missing required config: {name}")

    def get(self, path: str, default=None):
        """
        Dot-separated path lookup.
        e.g. get("memory.api.host") or get("runtime.heartbeat.interval_seconds")
        """
        keys = path.split(".")
        value = self.configs

        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]

        return value
