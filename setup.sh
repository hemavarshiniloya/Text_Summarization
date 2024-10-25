from setuptools import setup, find_packages

setup(
    name="text-summarizer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'streamlit>=1.24.0',
        'nltk>=3.8.1',
        'numpy>=1.24.3',
        'networkx>=3.1',
        'scikit-learn>=1.2.2',
        'requests>=2.31.0',
        'beautifulsoup4>=4.12.2',
        'PyPDF2>=3.0.1',
        'python-docx>=0.8.11',
        'scipy>=1.10.1',
    ],
    author="Hema Varshini Loya",
    author_email="hemavarshiniloya@gmail.com",
    description="A text summarization application with multiple methods",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/text-summarizer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
