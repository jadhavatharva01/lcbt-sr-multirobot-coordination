from setuptools import find_packages, setup

package_name = 'sar_world_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jadhavatharva01',
    maintainer_email='jadhavatharva01@todo.todo',
    description='World simulation helpers',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'grasp_zone_node = sar_world_sim.grasp_zone_node:main',
 	     'world_oracle_node = sar_world_sim.world_oracle_node:main',
            'metrics_logger_node = sar_world_sim.metrics_logger_node:main',


        ],
    },
)
