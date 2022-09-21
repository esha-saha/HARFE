from setuptools import setup
import io

VERSION = '0.0.1'

def main():
    with io.open("README.md", encoding="utf8") as f:
        long_description = f.read().strip()

    setup(
        name='harfe',
        url='https://github.com/esaha2703/HARFE',
        author='Esha Saha',
        author_email='esaha@uwaterloo.ca',
        packages=['harfe'],
        install_requires=[
            "numpy",
        ],
        version=VERSION,
        python_requires=">=3.6",
        license='MIT',
        description='A Python implimentation of Hard Ridge Random Feature Expansion',
        long_description=long_description,
        long_description_content_type="text/markdown",
    )

if __name__ == '__main__':
    main()
