from tqdm.auto import tqdm

from popsynth.utils.configuration import popsynth_config


def progress_bar(itr, **kwargs):

    return (tqdm(itr, **kwargs) if popsynth_config.show_progress else itr)
