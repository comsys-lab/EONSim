from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "lru_cache",
        ["lru_cache.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11']
    ),
]

setup(
    name="lru_cache",
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4.3'],
)
