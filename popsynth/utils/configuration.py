import os
import warnings
from pathlib import Path
from dataclasses import dataclass

from omegaconf import OmegaConf

_config_path = Path("~/.config/popsynth/").expanduser()

_config_name = Path("popsynth_config.yml")

_config_file = _config_path / _config_name


# Define structure with dataclasses
@dataclass
class LogConsole:

    on: bool = True
    level: str = "WARNING"


@dataclass
class LogFile:

    on: bool = True
    level: str = "WARNING"


@dataclass
class Logging:

    debug: bool = False
    console: LogConsole = LogConsole()
    file: LogFile = LogFile()


@dataclass
class Cosmology:

    Om: float = 0.307
    h0: float = 67.7


@dataclass
class PopSynthConfig:

    logging: Logging = Logging()
    cosmology: Cosmology = Cosmology()
    show_progress: bool = True


# Read the default config
popsynth_config: PopSynthConfig = OmegaConf.structured(PopSynthConfig)

# Merge with local config
if _config_file.is_file():

    _local_config = OmegaConf.load(_config_file)

    popsynth_config: PopSynthConfig = OmegaConf.merge(popsynth_config,
                                                      _local_config)

# Write defaults
else:

    # Make directory if needed
    _config_path.mkdir(parents=True, exist_ok=True)

    with _config_file.open("w") as f:

        OmegaConf.save(config=popsynth_config, f=f.name)
