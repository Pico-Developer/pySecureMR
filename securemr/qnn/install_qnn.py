# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Install qnn toolchain."""
from pathlib import Path
import os
from ..core.utils import run


def download_if_need(qnn_version="2.29.0.241129"):
    """Download qnn zip from remote."""
    QNN_SDK_ROOT = os.getenv("QNN_SDK_ROOT", Path(f"~/opt/qairt/{qnn_version}").expanduser())
    os.environ["QNN_SDK_ROOT"] = str(QNN_SDK_ROOT)
    if QNN_SDK_ROOT.exists():
        print(f"{QNN_SDK_ROOT} exists, skip download")
        return
    target = Path("~/opt").expanduser()
    target.mkdir(exist_ok=True)
    zipfile = f"https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v{qnn_version}.zip"
    cmd = f"cd /tmp; wget {zipfie}; cd {target}; unzip /tmp/v{qnn_version}.zip"
    run(cmd)
    assert QNN_SDK_ROOT.exists()


def install_deps():
    QNN_SDK_ROOT = os.getenv("QNN_SDK_ROOT")
    cmd = f"sudo {QNN_SDK_ROOT}/bin/check-linux-dependency.sh"
    run(cmd)
    cmd = f"sudo apt-get update && sudo apt-get install python3.10 python3-distutils libpython3.10"
    run(cmd)
    cmd = f"cd {QNN_SDK_ROOT}/bin; source ./envsetup.sh;  python3 {QNN_SDK_ROOT}/bin/check-python-dependency"
    run(cmd)
    
    # # install clang-14
    # source $SCRIPTS/install/install_clang.sh
    # install_type3 14
    # ${QNN_SDK_ROOT}/bin/envcheck -c

    # # install ndk 
    # bash $SCRIPTS/install/install_ndk.sh


if __name__ == "__main__":
    download_if_need()
    install_deps()
