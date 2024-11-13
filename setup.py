from setuptools import setup, find_packages

setup(
    name='tinystyler',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'sentencepiece',
        'sentence-transformers',
        'einops',
        'mutual-implication-score',
        'torch',
        'protobuf',
    ],  # These are minimal requirements, run `pip install -r requirements.txt`` for data generation and training requirements
)
