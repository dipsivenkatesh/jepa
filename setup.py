from setuptools import setup, find_packages

setup(
    name="openjepa",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "tqdm",
        "datasets",
        "wandb",
        "scikit-learn",
    ],
    entry_points={
        'console_scripts': [
            'openjepa-train=cli.train:main',
            'openjepa-evaluate=cli.evaluate:main'
        ]
    },
)