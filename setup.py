import setuptools
import pathlib


setuptools.setup(
    name='gaifo',
    version='1.0.0',
    description='Mastering Atari with Discrete World Models',
    url='http://github.com/RuiHirano/gaifo',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=['gaifo'],
    package_data={'gaifo': ['configs.yaml']},
    entry_points={'console_scripts': ['gaifo=gaifo.train:main']},
    install_requires=[
        'gym[atari]', 'atari_py', 'crafter', 'dm_control', 'ruamel.yaml',
        'tensorflow', 'tensorflow_probability'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
