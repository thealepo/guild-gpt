from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True, kw_only=True)
class GuildGPTConfig:
    