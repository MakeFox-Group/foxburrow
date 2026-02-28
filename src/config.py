"""INI config parser for foxburrow.ini."""

import configparser
import os
from dataclasses import dataclass, field


@dataclass
class GpuConfig:
    uuid: str = ""
    name: str = ""
    enabled: bool = True
    capabilities: set[str] = field(default_factory=set)
    onload: set[str] = field(default_factory=set)        # pre-load at startup
    unevictable: set[str] = field(default_factory=set)    # never evict from VRAM


@dataclass
class ServerConfig:
    address: str = "0.0.0.0"
    port: int = 8800
    models_dir: str = "models/"
    tensorrt_cache: str = "data/trt_cache/"
    enabled: bool = False
    secret: str = ""
    hf_token: str = ""  # HuggingFace API token for faster downloads


@dataclass
class SchedulerConfig:
    starvation_linear_s: float = 30.0    # Phase 1 duration (linear ramp)
    starvation_hard_s: float = 90.0      # Phase 2 ends → hard override
    load_rate_mb_s: float = 500.0        # Model load rate for time-to-ready (MB/s)
    status_push_interval_s: float = 2.0  # WebSocket status push interval


@dataclass
class ThreadsConfig:
    fingerprint: int = 0  # 0 = auto: min(8, cpu_count - 1)


def _auto_threads(configured: int, default_max: int) -> int:
    """Resolve thread count: 0 means auto-detect."""
    if configured > 0:
        return configured
    cpus = os.cpu_count() or 4
    return min(default_max, max(cpus - 1, 1))


@dataclass
class FoxBurrowConfig:
    server: ServerConfig
    gpus: list[GpuConfig]
    threads: ThreadsConfig = field(default_factory=ThreadsConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @staticmethod
    def load_from_file(path: str) -> "FoxBurrowConfig":
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        parser = configparser.ConfigParser()
        with open(path, encoding="utf-8") as f:
            parser.read_file(f)

        # Parse [server] section
        server = ServerConfig()
        if parser.has_section("server"):
            s = parser["server"]
            server.address = s.get("address", server.address)
            server.port = s.getint("port", server.port)
            server.models_dir = s.get("models_dir", server.models_dir)
            server.tensorrt_cache = s.get("tensorrt_cache", server.tensorrt_cache)
            server.enabled = s.getboolean("enabled", server.enabled)
            server.secret = s.get("secret", server.secret)
            server.hf_token = s.get("hf_token", server.hf_token)

        # Parse [scheduler] section
        scheduler = SchedulerConfig()
        if parser.has_section("scheduler"):
            s = parser["scheduler"]
            scheduler.starvation_linear_s = s.getfloat("starvation_linear_s", scheduler.starvation_linear_s)
            scheduler.starvation_hard_s = s.getfloat("starvation_hard_s", scheduler.starvation_hard_s)
            scheduler.load_rate_mb_s = s.getfloat("load_rate_mb_s", scheduler.load_rate_mb_s)
            scheduler.status_push_interval_s = s.getfloat("status_push_interval_s", scheduler.status_push_interval_s)

        # Parse [threads] section
        threads = ThreadsConfig()
        if parser.has_section("threads"):
            t = parser["threads"]
            threads.fingerprint = t.getint("fingerprint", threads.fingerprint)

        # Parse [GPU-<uuid>] sections
        gpus: list[GpuConfig] = []
        for section in parser.sections():
            if section.upper().startswith("GPU-") and len(section) > 4:
                uuid = section  # Preserve full section name as UUID key
                gpu = GpuConfig(uuid=uuid)
                gpu.enabled = parser.getboolean(section, "enabled", fallback=True)
                gpu.name = parser.get(section, "name", fallback="")
                caps_str = parser.get(section, "capabilities", fallback="")
                gpu.capabilities = {
                    c.strip().lower()
                    for c in caps_str.split(",")
                    if c.strip()
                }
                onload_str = parser.get(section, "onload", fallback="")
                gpu.onload = {
                    c.strip().lower()
                    for c in onload_str.split(",")
                    if c.strip()
                }
                unevictable_str = parser.get(section, "unevictable", fallback="")
                gpu.unevictable = {
                    c.strip().lower()
                    for c in unevictable_str.split(",")
                    if c.strip()
                }
                gpus.append(gpu)

        return FoxBurrowConfig(server=server, gpus=gpus, threads=threads, scheduler=scheduler)
