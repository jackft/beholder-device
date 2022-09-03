import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="beholder-recorder",
    version="0.0.1",
    author="Jack Terwilliger",
    author_email="jack.f.terwilliger@gmail.com",
    description="Controller for gstreamer video recorder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    packages=[
        "beholder.recorder",
        "beholder.webapp",
        "tests"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={
        "beholder.recorder": ["py.typed"],
        "beholder.webapp": ["py.typed"],
    },
    python_requires='>=3.8',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)