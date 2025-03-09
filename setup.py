from setuptools import setup, find_packages  # type: ignore

setup(
    name="tfsage",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    include_package_data=True,
    package_data={
        "tfsage": ["assets/*"],
    },
)

# pip install -e .
