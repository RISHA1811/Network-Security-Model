'''
ye setup.py esliye banate ha taaki hum requirements.txt jo hoga usko easily download kar saku
'''

from setuptools import find_packages, setup ## Ye saare files ko scan karega ur jaha __ini__.py hoga usko ek module banayega
from typing import List

def get_requirements()->List[str]:
    '''
    This Function will return the list of requirements 
    '''
    requirement_lst:List[str]=[]

    try:
        with open('requirements.txt') as file:
            ## Read Lines from the file 
            lines=file.readlines()
            for line in lines:
                requirement=line.strip()
                ## Ignore the empty lines and -e
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt not found")




setup(
    name='Network-Security',
    version='0.0.1',
    author='Rishabh Jha',
    author_email='rishabhjha1811@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)

