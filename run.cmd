if exist ./exllama (
    echo "exllama exists"
) else (
    echo "exllama does not exist"
    git clone https://github.com/turboderp/exllama
    git reset --hard 82369c5
)
if exist ./venv (
    echo "venv exists"
) else (
    echo "venv does not exist"
    python -m venv ./venv
)


set PY="%cd%\venv\Scripts\python.exe"
set PIP="%cd%\venv\Scripts\pip.exe"

:: install pytorch with cuda
call %PY% -m pip install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu117

:: install Ninja
call %PY% -m pip install ninja

:: install requirements
call %PY% -m pip install -r requirements.txt
call %PY% main.py
