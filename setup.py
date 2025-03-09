from setuptools import setup, find_packages

setup(
    name="tfsage",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    include_package_data=True,
    package_data={
        "tfsage": ["assets/*"],
        "tfsage.embedding": ["seurat_integration.R"],
    },
)

# pip install -e .
