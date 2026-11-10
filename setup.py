import setuptools

setuptools.setup(
    name="SlowFast",
    version="0.0.1",
    author="DigiNova",
    author_email='info@diginova.com.tr',
    description="Video action recognition with the SlowFast network (PyTorchVideo, Kinetics-400)",
    url='https://github.com/Zerothlaw780/slowfast',
    license='MIT',
    install_requires=['sdk', 'opencv-python-headless', 'numpy', 'torch', 'pytorchvideo==0.1.5'],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    packages=[
        'novavision.SlowFast',
        'novavision.SlowFast.executors',
        'novavision.SlowFast.models',
        'novavision.SlowFast.utils',
    ],
    package_dir={'novavision.SlowFast': 'src'},
    python_requires=">=3.6"
)
