from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path):
    requirements=[]
    with open(file_path) as f_p:
        requirements=f_p.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        return requirements


setup(
    name='Diamond_Price_Prediction',
    version='0.0.1',
    author='surya',
    author_email='suryakadali1994@gmail.com',
    install_requires=get_requirements('/Users/suryasaikadali/Downloads/pw_skills/end_to_End_projects/Diamond_price_prediction/requirements.txt'),
    packages=find_packages()

)