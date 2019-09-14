#!/bin/sh

###### configure rpath ######
# python deployConfig.py

###### clean up existing packages ######
rm hdff/dist/*
rm hdtopology/dist/*
rm ngl/dist/*

###### packaging for PyPI ######
cd ngl
python3 setup.py sdist bdist_wheel
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ --verbose dist/*

cd ../hdff
python3 setup.py sdist bdist_wheel
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ --verbose dist/*

cd ../hdtopology
python3 setup.py sdist bdist_wheel
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ --verbose dist/*
