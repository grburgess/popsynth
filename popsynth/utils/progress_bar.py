from tqdm.auto import tqdm

from popsynth.utils.configuration import popsynth_config

_purple = "#B833FF"


def progress_bar(itr, **kwargs):

    return (tqdm(itr, colour=_purple, **kwargs)
            if popsynth_config.show_progress else itr)
