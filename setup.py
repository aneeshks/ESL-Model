from setuptools import setup, find_packages

with open('README.rst', 'r', encoding='utf8') as f:
    readme = f.read()


version = __import__('esl_model').__version__


setup(name='esl_model',
      version=version,
      description="Algorithm from The Elements of Statistical Learning book implement by Python 3 code",
      long_description=readme,
      classifiers=[
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3 :: Only',
          'Development Status :: 2 - Pre-Alpha',
          'License :: OSI Approved :: MIT License',

      ],
      url='https://github.com/littlezz/ESL-Model',
      author='littlezz',
      author_email='zz.at.field@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['tests*']),
      install_requires=[
            'numpy',
            'pandas',
            'scipy',
            'scikit-learn',
      ],
      tests_require=['pytest'],
      include_package_data=True,

      zip_safe=False,


)