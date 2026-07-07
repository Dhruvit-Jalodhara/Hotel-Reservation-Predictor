from setuptools import find_packages , setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path : str) -> List[str]: 
    '''
    this function will return the list of requirements 

    parameters :
    -----------
        file_path : str -> path of file containing names of requirements models 

    return :
    ----------
        requirements : list -> contaings all requirement modules
    '''
    requirements = []
    with open(file_path) as file_obj :
        # .strip() removes newlines AND hidden spaces
        requirements = [req.strip() for req in file_obj.readlines()]
        
        # Remove any empty lines that might have been in the text file
        requirements = [req for req in requirements if req]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name = 'Hotel Reservation Prediction' ,
    version= '0.1.0' ,
    author= 'Dhruvit Jalodhara',
    packages= find_packages(),
    install_requires= get_requirements('requirements.txt')
)