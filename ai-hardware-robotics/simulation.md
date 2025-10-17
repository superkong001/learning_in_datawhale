论文地址

https://drive.google.com/file/d/1637XPqWMajfC_ZqKfCGxDxzRMrsJQA1g/view

项目网站[DISCOVERSE](https://air-discoverse.github.io/)

## 📦 Installation 📦安装

[](https://github.com/TATP-233/DISCOVERSE/tree/main#-installation)

```shell
## Clone repository
git clone https://github.com/TATP-233/DISCOVERSE.git --recursive
cd DISCOVERSE

## Choose installation method
conda create -n AI_simulation python=3.10 -y
conda activate AI_simulation
pip install -e .

## Auto-detect and download required submodules
python scripts/setup_submodules.py

## Verify installation
python scripts/check_installation.py

## Installation by Use Case
## Scenario 1: Learning Robot Simulation Basics
pip install -e .  # Core functionality only


```
