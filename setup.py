from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import shutil

class PostInstallCommand(install):
    def run(self):
        install.run(self)
        current_dir = os.getcwd()
        source_model_path = os.path.join(os.path.dirname(__file__), 'SentiAnalyzer', 'SentiToolKit.keras')
        source_tokenizer_path = os.path.join(os.path.dirname(__file__), 'SentiAnalyzer', 'tokenizer.pkl')
        shutil.copy(source_model_path, current_dir)
        shutil.copy(source_tokenizer_path, current_dir)
        print("Copied SentiToolKit.keras and tokenizer.pkl to the project directory.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SentiToolKit",
    version="0.1.7", 
    packages=find_packages(),
    package_data={
        'SentiAnalyzer': ['SentiToolKit.keras', 'tokenizer.pkl'], 
    },
    install_requires=[
        "tensorflow>=2.0",
        "scikit-learn",
    ],
    author="Niall Dcunha",
    author_email="dcunhaniall@gmail.com",
    description="A sentiment analysis toolkit using a trained TensorFlow model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Niall1985/SentiToolKit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'sentitoolkit=SentiAnalyzer.Main_tensor_model:main',
        ],
    },
    cmdclass={
        'install': PostInstallCommand,
    },
)
