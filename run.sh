if [ -d "./exllama" ]
then
    echo "exllama exists"
else
    echo "exllama does not exist"
    git clone https://github.com/turboderp/exllama
    git reset --hard 82369c5
fi

if [ -d "./venv" ]
then
    echo "venv exists"
else
    echo "venv does not exist"
    python3 -m venv ./venv
fi

# set venv python location to var
PY="$(pwd)/venv/bin/python"

# set venv pip location to var
PIP="$(pwd)/venv/bin/pip"

# install pytorch with cuda
$PIP install torch torchvision torchaudio --upgrade -f https://download.pytorch.org/whl/cu117

# install requirements
$PIP install -r requirements.txt

# run the Python program
$PY main.py