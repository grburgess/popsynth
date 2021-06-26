from pathlib import Path
import pkg_resources


def get_path_of_data_file(data_file) -> Path:
    file_path = pkg_resources.resource_filename("popsynth",
                                                "data/%s" % data_file)

    return Path(file_path)


def get_path_of_log_dir() -> Path:

    p: Path = Path("~/.log/popsynth").expanduser()

    if not p.exists():

        p.mkdir(parents=True)

    return p


def get_path_of_log_file(file_name: str) -> Path:

    return get_path_of_log_dir() / file_name
