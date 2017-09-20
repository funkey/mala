from setuptools import setup

setup(
        name='mala',
        version='0.2',
        description='Training and evaluation scripts for MALA (https://arxiv.org/abs/1709.02974).',
        url='https://github.com/funkey/mala',
        author='Jan Funke',
        author_email='jfunke@iri.upc.edu',
        license='MIT',
        packages=[
            'mala',
            'mala.networks',
        ],
        install_requires=[
            "tensorflow",
        ]
)
