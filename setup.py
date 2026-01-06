import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

try:
    with open("requirements.linux.txt", "r", encoding="utf-8") as f:
        install_requires = f.read().splitlines()
except FileNotFoundError:
    print("Warning: requirements.linux.txt not found. Package may not have all dependencies listed.")
    install_requires = []

install_requires = [req for req in install_requires if req and not req.startswith(
    '#') and not req.startswith('[source')]


setuptools.setup(
    name="attention-survey",
    version="0.1.0",
    author="ghubnerr, davidulloa6310",
    author_email="gabrielhubnerlucchesi@gmail.com, dulloa6310@gmail.com",
    description="A survey and implementation of state-of-the-art attention mechanisms in Flax/JAX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ghubnerr/attention-mechanisms",
    packages=setuptools.find_packages(
        exclude=['tests', 'tests.*', 'notebooks', 'notebooks.*']
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.9',
    install_requires=install_requires,
    keywords="attention, transformer, flax, jax, mha, gqa, mqa, mla, moe, rope",
)
