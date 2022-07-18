import setuptools

setuptools.setup(
    name="paper-candy",
    version="0.0.1b3",
    author="ATATC",
    author_email="futerry@outlook.com",
    description="A loosely coupled training framework for Deep Learning research.",
    license='Apache License 2.0',
    long_description="**PaperCandy** is a loosely coupled training framework for Deep Learning research. "
                     "It provides an intermediate layer so that the back-end frameworks(such as PyTorch or TensorFlow) "
                     "can be replaced easily.",
    long_description_content_type="text/markdown",
    url="https://github.com/ATATC/PaperCandy",
    packages=setuptools.find_packages(),
    install_requires=["torch", "opencv-python"],
)
