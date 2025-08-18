from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(filename: str) -> List[str]:
    requirements = []

    with open(filename, 'r') as f:
        requirements = f.readlines()

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        requirements = [req.replace("\n", "") for req in requirements]
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    author='Srinu',
    author_email='srinunayakk7@gmail.com',
    install_requires=get_requirements('requirements.txt'),
)