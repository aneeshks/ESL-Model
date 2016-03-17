from setuptools import setup

with open('README.rst', 'r', encoding='utf8') as f:
    readme = f.read()


version = __import__('esl_model').__version__

setup(name='esl_model',
      version=version,
      description="Algorithm from The Elements of Statistical Learning book implement by Python code",
      url=''

)