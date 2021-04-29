import setuptools
import sys

current_platform = sys.platform.lower()
print(current_platform)
is_windows = current_platform.startswith('win')
is_ubuntu = 'linux' in current_platform
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
    name="ngl",
    version="0.0.1",
    author="Shusen Liu, Peer-Timo Bremer",
    author_email="liu42@llnl.gov, bremer5@llnl.gov",
    description="A wrapper library for the C++ NGL library (http://www.ngraph.org/)",
    install_requires=['numpy==1.19'],
    keywords="Empty Region Graph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),

    include_package_data=True,
    package_data={'': binaryInstallPostfix},
    license="BSD-3",

    cmdclass={'bdist_wheel': bdist_wheel},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: BSD License"
    ],
)
