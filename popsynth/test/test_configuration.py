import tempfile
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from popsynth.utils.configuration import PopSynthConfig


def test_default_configuration():
    """
    Just load the default config.
    """

    default_config = PopSynthConfig()


def test_configuration_write():
    """
    Save config locally and reload.
    """

    popsynth_config = PopSynthConfig()

    with tempfile.NamedTemporaryFile() as f:

        OmegaConf.save(config=popsynth_config, f=f.name)

        loaded_config = OmegaConf.load(f.name)

    assert popsynth_config == loaded_config


def test_user_config_merge():
    """
    Make partial user configs and merge.
    """

    popsynth_config = PopSynthConfig()

    user_configs = [
        {
            "show_progress": False
        },
        {
            "logging": {
                "console": {
                    "on": False,
                    "level": "INFO"
                }
            }
        },
    ]

    for i, config in enumerate(user_configs):

        path = Path(f"config_{i}.yml")

        with path.open("w") as f:

            yaml.dump(stream=f, data=config, Dumper=yaml.SafeDumper)

        loaded_config = OmegaConf.load(path)

        popsynth_config = OmegaConf.merge(popsynth_config, loaded_config)

        path.unlink()

    assert not popsynth_config.show_progress

    assert not popsynth_config["logging"]["console"]["on"]

    assert popsynth_config["logging"]["console"]["level"] == "INFO"
