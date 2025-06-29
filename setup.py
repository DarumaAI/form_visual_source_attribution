from setuptools import find_packages, setup

setup(
    name="form_visual_source_attribution",  # Replace with your module/package name
    version="0.1.0",
    author="DarumaAI",
    author_email="daruma.ai.phd@gmail.com",
    description="Aligns multimodal embeddings with text and image inputs for visual source attribution.",
    url="https://github.com/DarumaAI/form_visual_source_attribution.git",  # Optional
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
