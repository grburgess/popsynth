from setuptools import setup
import os

import versioneer


# Create list of data files
def find_data_files(directory):

    paths = []

    for (path, directories, filenames) in os.walk(directory):

        for filename in filenames:

            paths.append(os.path.join("..", path, filename))

    return paths


extra_files = find_data_files("popsynth/data")

setup(
    version=versioneer.get_version(),
    include_package_data=True,
    package_data={"": extra_files},
    cmdclass=versioneer.get_cmdclass(),
)
