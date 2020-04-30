from setuptools import setup

setup(
   name='brainsprite',
   version='0.14dev0',
   description='Python API for the brainsprite javascript MRI brain viewer',
   author='Pierre Bellec and the brainsprite contributors',
   packages=['.'],  #same as name
   author_email='pierre.bellec@gmail.com',
   install_requires=['numpy', 'matplotlib', 'sklearn', 'nilearn', 'tempita'], #external packages as dependencies
)
