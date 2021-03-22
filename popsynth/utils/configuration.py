import os
import warnings
from pathlib import Path

from omegaconf import OmegaConf

# from configya import YAMLConfig

structure = {}

structure["logging"] = dict(
    debug=False,
    console=dict(on=True, level="WARNING"),
    file=dict(on=True, level="INFO"),
)
structure["cosmology"] = dict(Om=0.307, h0=67.7)
structure["show_progress"] = True


class NoConfigurationWarning(RuntimeWarning):
    pass


class PopSynthConfig:
    def __init__(
        self,
        structure=structure,
        config_path="~/.config/popsynth/",
        config_name="popsynth_config.yml",
    ) -> None:

        self._default_structure = structure

        self._config_path = Path(config_path).expanduser()

        self._make_config_dir_if_needed()

        self._config_name = config_name

        self._full_path = self._config_path / self._config_name

        if not self._config_file_exists():

            warnings.warn(
                f"No configuration file found! Making one in {self._full_path}",
                NoConfigurationWarning,
            )

            # Write
            self._config = OmegaConf.create(self._default_structure)
            OmegaConf.save(self._config, f=self._full_path)

        # Read
        self._config = OmegaConf.load(self._full_path)

    def _make_config_dir_if_needed(self):

        if not os.path.exists(self._config_path):

            os.makedirs(self._config_path)

    def _config_file_exists(self):

        return os.path.exists(self._full_path)

    @property
    def config(self):

        return self._config


popsynth_config = PopSynthConfig().config
