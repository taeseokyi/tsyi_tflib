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
    package_dir={'tsyi_tflib':'src'},
    package_data={'tsyi_tflib':['data/*.txt']},
    data_files=[('config', ['cfg1/text1.cfg', 'cfg2/text2.cfg'])],
    zip_safe=False,
    install_requires=[
        'numpy>=1.19.2'
    ]
)