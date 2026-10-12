"""Builds the Rust plugin with cargo and packages it, with its manifest, as a platform wheel.

The wheel holds no Python extension module, only a shared library that PyTorches loads itself, so it
is tagged `py3-none-<platform>` and works with any Python 3 interpreter.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT / "python" / "pytorches_cuda"
LIB_NAMES = ["pytorches_plugin_cuda.dll", "libpytorches_plugin_cuda.so", "libpytorches_plugin_cuda.dylib"]


def build_plugin():
    """Compile the plugin and place the library in the package directory (as pytorches_plugin_cuda.*)."""
    subprocess.run(["cargo", "build", "--release", "--lib", "--manifest-path", str(ROOT / "Cargo.toml")], check=True)
    target = ROOT / "target" / "release"
    for name in LIB_NAMES:
        built = target / name
        if built.exists():
            # The loader expects pytorches_plugin_<name>.<ext> with no "lib" prefix.
            shutil.copy2(built, PACKAGE_DIR / name.removeprefix("lib"))
            return
    sys.exit(f"cargo did not produce a plugin library in {target}")


class BuildPy(build_py):
    def run(self):
        if not any((PACKAGE_DIR / n.removeprefix("lib")).exists() for n in LIB_NAMES):
            build_plugin()
        super().run()


class BinaryDistribution(Distribution):
    def has_ext_modules(self):  # makes the wheel platform-specific
        return True


class BdistWheel(bdist_wheel):
    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"build_py": BuildPy, "bdist_wheel": BdistWheel})
