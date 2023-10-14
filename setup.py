import setuptools

with open("requirements.txt", "r") as file:
    requirements = file.read().splitlines()

setuptools.setup(
    name="food_smart",
    version="0.0.1",
    author="food_smart",
    author_email="food_smart",
    description="Knowing What You Eat: Health Indicator Analysis Through Ingredient Recognition",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.1,<4.0',
    install_requires=requirements,
    dependency_links=[],
)