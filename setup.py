from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    packages=find_packages(),
    rust_extensions=[
        RustExtension(
            "neurenix._phynexus",
            path="src/phynexus/rust/Cargo.toml",
            binding=Binding.PyO3,
            features=["python"],
        )
    ],
    zip_safe=False,
)
