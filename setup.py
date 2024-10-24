from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SentiToolKit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0",
        "scikit-learn",
    ],
    author="Niall Dcunha",
    author_email="dcunhaniall@gmail.com",
    description="A sentiment analysis toolkit using a pre-trained TensorFlow model",
    long_description=long_description,  
    long_description_content_type="text/markdown",
    url="https://github.com/Niall1985/SentiToolKit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU AFFERO GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'sentitoolkit=senti_toolkit:main',
        ],
    },
)
