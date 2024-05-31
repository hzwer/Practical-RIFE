from setuptools import setup


with open("requirements.txt") as f:
    required = f.read().splitlines()

if __name__ == "__main__":
    setup(
        name="practical_rife",
        package_data={
            "practical_rife": [
                "**/*.txt",
                "**/*.t7",
                "**/*.pth",
                "**/*.json",
                "**/*.yaml",
                "**/*.yml",
            ]
        },
    )
