from distutils.core import setup


with open("./requirements.txt", "r") as f:
    requirements = f.readlines()


setup(
    name="fractal_zero",
    version="0.0.1",
    author="Dyllan McCreary",
    description="Fractal MuZero",
    author_email="dyllanmccreary@gmail.com",
    packages=["fractal_zero"],
    install_requires=requirements,
)
