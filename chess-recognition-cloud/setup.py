from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='Chess recognition',
      author='Ivan Adamov',
      author_email='adamovivan@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py'
      ],
      zip_safe=False)
