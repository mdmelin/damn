from setuptools import setup, find_packages

setup(
    name="DAMN",
    author="Max Melin",
    author_email="mmelin@ucla.edu",
    description="DAMN (Design A Matrix Now) - tools for building and fitting design matrices and GLMS.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "tqdm",
        "matplotlib"
    ],
    packages=find_packages(where="."),
    package_dir={"": "."},
    include_package_data=True,
    #entry_points={
    #    "console_scripts": [
    #        "schemaviewer = ibl_schema.cli:website",
    #    ],
    #},
    url="https://github.com/mdmelin/DAMN",
    #project_urls={
    #    "Homepage": "https://github.com/mdmelin/personal_data_schema",
    #    "Issues": "https://github.com/mdmelin/personal_data_schema/issues",
    #},
    # version could be dynamically retrieved here if needed
    # version="0.0.1"  # Uncomment this or use the dynamic approach if necessary
)