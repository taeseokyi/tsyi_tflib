from gettext import install
from setuptools import setup, find_packages

setup(
    name='tsyi_tflib',
    version='0.0.0',
    description='tsyi pip install test',
    url='https://github.com/taeseokyi/tf-official-common.git',
    author='taeseok yi',
    author_email='tsyi@kisti.re.kr',
    license='taeseok yi',
    python_requires='>=3',
    include_data_package=True,
    packages=find_packages(exclude = []),
    package_data={'':['data/tech_name_tag/*.txt']},
    zip_safe=False,
    install_requires=[
        'numpy>=1.19.2'
    ]
)