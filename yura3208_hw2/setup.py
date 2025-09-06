from setuptools import find_packages, setup

package_name = 'yura3208_hw2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yusif',
    maintainer_email='yusifrazz@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher = yura3208_hw2.publisher:main',
            'subscriber = yura3208_hw2.subscriber:main',
            'service = yura3208_hw2.service:main',
            'client = yura3208_hw2.client:main',
        ],
    },
)
