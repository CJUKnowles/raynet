# !/bin/bash

# Automatically creates a Python virtual environment, and installs RayNet-critical dependencies.
# Should be run once, at installation, from the root of the repo.
# Alternatively you can use your own virtual environment, but make sure to install the dependencies in requirements.txt.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-extra.txt

echo ""
echo "----------------------------------------"
echo "Virtual environment installed. Make sure to activate it before running RayNet:"
echo -e "\tsource .venv/bin/activate"
echo "----------------------------------------"
echo ""