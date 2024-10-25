from setuptools import setup, find_packages

setup(
    name="text-summarizer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'nltk',
        'numpy',
        'networkx',
        'scikit-learn',
        'requests',
        'beautifulsoup4',
        'PyPDF2',
        'python-docx',
        'scipy',
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
