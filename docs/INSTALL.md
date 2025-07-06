# Installation

### Requirements
All the codes are tested in the following environment (working as of  Jul 2025):
* Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
* Python 3.10.18 
* add dead snakes repo so that your torch does not crash!!
* ```shell
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install python3.10 python3.10-venv
    ```
* torch will likely give you cuda uncompatibility error , so to silence that
* PyTorch 1.1 or higher (tested on PyTorch 1.1, 1,3, 1,5~1.10)
* Tested on CUDA 12.9 (Ada Love Lace Arch 4090) and works!
* Spconv version is mentioned in requirements.txt


### Install 

* No need of separatly doing anything I have taken care of spconv and other things...

a. Do the following:
```bash
python3.10 -m venv pvt
source pvt/bin/activate

pip install --upgrade pip

pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.1+cu118 \
  torchvision==0.22.1+cu118 \
  torchaudio==2.7.1+cu118

pip install \
  torch-scatter==2.1.2+pt27cu118 \
  -f https://data.pyg.org/whl/torch-2.7.0+cu118.html

pip install -r requirements.txt

pip install --upgrade pip setuptools wheel
```
from your home folder

Also on running this on cuda 12.9 for me I encountered cuda version issues which I slapped by commenting out:
```bash
${YOUR_VENV_FOLDER}$/lib/python3.10/site-packages/torch/utils/cpp_extension.py
```
or if you did exactly what I did and named it pvt
```bash
cd pvt/lib/python3.10/site-packages/torch/utils/
```
then edit cpp_extension.py

in that there will be a function which checks pytorch's compatibility with cuda version installed...
since nvidia drivers are backward compatible so we ignore such checks which are time wasters for our precious life
Anyway you will find this function `_check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:`
inside it comment out stuff like:
```python
    if cuda_ver != torch_cuda_version:
        # major/minor attributes are only available in setuptools>=49.4.0
        if getattr(cuda_ver, "major", None) is None:
            # raise ValueError("setuptools>=49.4.0 is required")
            pass
        if cuda_ver.major != torch_cuda_version.major:
            # raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
            pass
        # warnings.warn(CUDA_MISMATCH_WARN.format(cuda_str_version, torch.version.cuda))
```
b. In 
```shell
${YOUR_VENV_FOLDER}$/lib/python3.10/site-packages/setuptools/command
or 
pvt/lib/python3.10/site-packages/setuptools/command
in develop.py
```
Comment out the below mentioned lines so that you are not stuck in a loop...
```python
import site
import subprocess
import sys

from setuptools import Command
from setuptools.warnings import SetuptoolsDeprecationWarning


class develop(Command):
    """Set up package for development"""

    user_options = [
        ("install-dir=", "d", "install package to DIR"),
        ('no-deps', 'N', "don't install dependencies"),
        ('user', None, f"install in user site-package '{site.USER_SITE}'"),
        ('prefix=', None, "installation prefix"),
        ("index-url=", "i", "base URL of Python Package Index"),
    ]
    boolean_options = [
        'no-deps',
        'user',
    ]

    install_dir = None
    no_deps = False
    user = False
    prefix = None
    index_url = None

    def run(self):
        # cmd = (
        #     [sys.executable, '-m', 'pip', 'install', '-e', '.', ]
        #     + ['--target', self.install_dir] * bool(self.install_dir)
        #     + ['--no-deps'] * self.no_deps
        #     + ['--user'] * self.user
        #     + ['--prefix', self.prefix] * bool(self.prefix)
        #     + ['--index-url', self.index_url] * bool(self.index_url)
        # )
        # subprocess.check_call(cmd)
        pass

    def initialize_options(self):
        DevelopDeprecationWarning.emit()

    def finalize_options(self) -> None:
        pass


class DevelopDeprecationWarning(SetuptoolsDeprecationWarning):
    _SUMMARY = "develop command is deprecated."
    _DETAILS = """
    Please avoid running ``setup.py`` and ``develop``.
    Instead, use standards-based tools like pip or uv.
    """
    _SEE_URL = "https://github.com/pypa/setuptools/issues/917"
    _DUE_DATE = 2025, 10, 31
```

c. Install this `pcdet` library and its dependent libraries by running the following command:
( please return to your original folder after doing above steps and then run it )
```shell
pip install -e . --no-build-isolation
```
d. Since Majority running code exists in tools/ , you may get the pcdet not found error , that comes from subdirs not being able to find specific python env code required for running the scripts <br>
Simply export the enviornment variable so that it becomes accessible...
```bash
export PYTHONPATH={location_of_your_PVT_SSD_folder}/:$PYTHONPATH
in my case it was
~/Documents/PVT-SSD
so the command becomes
export PYTHONPATH=~/Documents/PVT-SSD/:$PYTHONPATH
```

e. Enjoy!! (Remember to undo changes after you are happy , I mostly dont undo things since I want my env not to be upgraded since it creates conflicts... so you can either leave the commented out portions as it is or just uncomment them back. But if you uncomment `_check_cuda_version` then you wont be able to run...but for `develop.py` after installing `pcet` you are good to uncomment it out..)
