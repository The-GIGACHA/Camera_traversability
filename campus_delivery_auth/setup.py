## catkin Python 패키지 설치 설정 — setup.py 직접 실행 금지, catkin_python_setup()이 사용
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['campus_delivery_auth'],
    package_dir={'': 'src'},
)

setup(**setup_args)
