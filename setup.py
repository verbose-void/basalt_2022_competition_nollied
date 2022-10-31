from distutils.core import setup


with open("./requirements.txt", "r") as f:
    requirements = f.readlines()


setup(
    name="xirl_zero",
    version="0.0.1",
    author="Dyllan McCreary",
    description="",
    author_email="dyllanmccreary@gmail.com",
    packages=["xirl_zero", "fgz", "new_fgz"],
    install_requires=requirements,
)
