from setuptools import setup

setup(name='ENN',
      version='0.1',
      description='Python implementation of EXAMM and NEAT',
      author='Peter Grimm',
      author_email='peterdgrimm@gmail.com',
      license='MIT',
      packages=['ENN'],
      install_requires=[
          'numpy', 'random','math', 'itertools','threading','multiprocessing','enum','dill','cProfile','threading','importlib','pathos'
      ],
      zip_safe=False)
