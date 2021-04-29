import setuptools
import sys

current_platform = sys.platform.lower()
is_windows = current_platform.startswith('win')
is_ubuntu = 'ubuntu' in current_platform
is_mac = 'darwin' in current_platform

with open("README.md", "r") as fh:
    long_description = fh.read()

binaryInstallPostfix = []
if is_mac:
    binaryInstallPostfix.append('*.so')
    binaryInstallPostfix.append('*.dylib')
elif is_ubuntu:
    binaryInstallPostfix.append('*.so')
elif is_windows:
    binaryInstallPostfix.append('*.dll')
    binaryInstallPostfix.append('*.pyd')
else:
    binaryInstallPostfix.append('*.so')

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

setuptools.setup(
    name="hdff",
    version="0.0.1",
    author="Shusen Liu, Peer-Timo Bremer",
    author_email="liu42@llnl.gov, bremer5@llnl.gov",
    description="File Format Library for NDDAV System",
    url="https://github.com/LLNL/hdtopology",
    keywords="File Format, High-Dimensional Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    license="BSD-3",
    package_data={'': binaryInstallPostfix},
    cmdclass={'bdist_wheel': bdist_wheel},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: BSD License"
    ],
)
