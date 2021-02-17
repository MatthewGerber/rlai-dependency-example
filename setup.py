from setuptools import (
    setup, find_namespace_packages
)

INSTALL_REQUIREMENTS = [
    'rlai==0.17.0'
]

setup(
    name='rlai-dependency-example',
    version='0.0.0',
    description='An example of using the rlai package.',
    author='Matthew Gerber',
    author_email='gerber.matthew@gmail.com',
    url='https://matthewgerber.github.io/rlai',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='==3.8.5',
    install_requires=[
        INSTALL_REQUIREMENTS
    ]
)
