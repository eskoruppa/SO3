from setuptools import setup

setup(name='SO3',
      version='0.0.1',
      description='A collection of methods for the rotation group',
      url='https://github.com/eskoruppa/SO3',
      author='Enrico Skoruppa',
      author_email='enrico dot skoruppa at gmail dot com',
      license='GNU2',
      packages=['so3'],
      package_dir={
            'so3': 'so3',
      },
      zip_safe=False) 
