#!/bin/sh

###### configure rpath ######
# python deployConfig.py

###### clean up existing packages ######
rm -f hdff/dist/*
rm -f hdtopology/dist/*
rm -f ngl/dist/*

###### packaging for PyPI ######
cd hdtopology
python3 setup.py sdist bdist_wheel
for wheel in $(find . -iname "*.whl") ; do 
  mv $wheel $(echo $wheel | sed 's/-linux_/-manylinux1_/')
done
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ --verbose dist/*
cd ..
exit 1

cd ngl
python3 setup.py sdist bdist_wheel
for wheel in $(find . -iname "*.whl") ; do 
  mv $wheel $(echo $wheel | sed 's/-linux_/-manylinux1_/')
done
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ --verbose dist/*
cd ..

cd hdff
python3 setup.py sdist bdist_wheel
for wheel in $(find . -iname "*.whl") ; do 
  mv $wheel $(echo $wheel | sed 's/-linux_/-manylinux1_/')
done
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ --verbose dist/*
cd ..

