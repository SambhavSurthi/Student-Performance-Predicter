from setuptools import setup,find_packages
from typing import List

FILE_PATH='requirements.txt'
HYPHEN_E_DOT='-e .'

def get_requirements(filepath:str)->List[str]:
    '''this function will return the list of requirements'''
    requirements=[]
    with open(filepath) as fileObj:
        requirements=fileObj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
setup(
    name='Student-Performance-Predicter',
    version='0.0.1',
    description='This is a End To End Machine learning Project',
    author='Sambhav Surthi',
    author_email='sambhavsurthi.ai@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(filepath=FILE_PATH)
)