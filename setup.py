from setuptools import setup
import io

VERSION = '0.0.1'

def main():
    with io.open("README.md", encoding="utf8") as f:
        long_description = f.read().strip()

    with open("requirements.txt") as f:
            required = f.read().splitlines()

    setup(
        name='harfe',
        url='https://github.com',
        author='Esha Saha',
        author_email='esaha@uwaterloo.ca',
        packages=['harfe'],
        install_requires=required,
        version=VERSION,
        license='MIT',
        description='A Python implimentation of Hard Ridge Random Feature Expansion',
        long_description=long_description,
        long_description_content_type="text/markdown",
    )

if __name__ == '__main__':
    main()
