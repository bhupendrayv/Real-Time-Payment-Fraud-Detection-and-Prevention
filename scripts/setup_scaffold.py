import pathlib

dirs = [
    'data',
    'notebooks',
    'ml/artifacts',
    'api/tests',
    'simulator',
    'frontend/src',
    'monitoring',
    'docs',
    '.github/workflows',
]

for d in dirs:
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

keep_dirs = [
    'ml/artifacts',
    'api/tests',
    'simulator',
    'frontend/src',
    'monitoring',
    'docs',
    '.github/workflows',
]

for d in keep_dirs:
    pathlib.Path(d, '.gitkeep').touch(exist_ok=True)

print("Scaffolding created successfully.")
