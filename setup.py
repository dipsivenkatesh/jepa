from setuptools import setup, find_packages

setup(
    name="jepa",
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
            'jepa-train=cli.train:main',
            'jepa-evaluate=cli.evaluate:main'
        ]
    },
)