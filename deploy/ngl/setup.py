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
elif is_ubuntu:
    binaryInstallPostfix.append('*.so')
elif is_windows:
    binaryInstallPostfix.append('*.dll')
else:
    binaryInstallPostfix.append('*.so')


setuptools.setup(
    name="ngl",
    version="0.0.1",
    author="Shusen Liu, Peer-Timo Bremer",
    author_email="liu42@llnl.gov, bremer5@llnl.gov",
    description="A wrapper library for the C++ NGL library (http://www.ngraph.org/)",
    keywords="Empty Region Graph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    license="BSD-3",
    package_data={'': binaryInstallPostfix},

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: BSD License"
    ],
)
