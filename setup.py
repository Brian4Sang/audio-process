from setuptools import setup, find_packages

setup(
    name='audio_preprocess',
    version='0.1',
    description='Audio preprocessing pipeline for TTS training data.',
    packages=find_packages(),
    install_requires=[
        'torchaudio',
        'soundfile',
        'numpy',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [],
    },
)
