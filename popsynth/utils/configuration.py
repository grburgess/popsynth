from pathlib import Path

from configya import YAMLConfig

structure = {}

structure["logging"] = dict(debug=False,
                            console=dict(on=True, level="WARNING"),
                            file=dict(on=True, level="INFO"))
structure["cosmology"] = dict(Om=0.307, h0=67.7)
structure["show_progress"] = True

class PopSynthConfig(YAMLConfig):
    def __init__(self) -> None:

        super(PopSynthConfig, self).__init__(
            structure=structure,
            config_path="~/.config/popsynth/",
            config_name="popsynth_config.yml",
        )


popsynth_config = PopSynthConfig()
