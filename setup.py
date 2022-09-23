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
        'numpy>=1.19.2',
        'fastprogress==1.0.2',
        'seqeval==1.2.2',
        'tensorflow>=2.4.0',
        'transformers>=4.15.0',
        'sentencepiece==0.1.95',
        'mecab-ko==1.0.0',
        'mecab-ko-dic==1.0.0',
    ]
)