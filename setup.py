from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(filepath: str) -> List[str]:
    requirements = []

    with open(filepath) as f:
        requirements = f.readlines()

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        requirements = [req.replace("\n", "") for req in requirements]
    return requirements


setup(
    name = "ml_project",
    packages = find_packages(),
    version = "0.0.1",
    author = "Srinu",
    author_email = "srinunayakk9@gmail.com",
    install_requires = get_requirements("requirements.txt"),
    include_package_data = True,
)



