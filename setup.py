from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    packages=find_packages(include=['neurenix', 'neurenix.*']),
    rust_extensions=[
        RustExtension(
            "neurenix._phynexus",
            path="src/phynexus/rust/Cargo.toml",
            binding=Binding.PyO3,
            features=["python"],
        )
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'neurenix=neurenix.cli.cli:main',
        ],
    },
)
