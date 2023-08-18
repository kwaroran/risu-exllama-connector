import uvicorn
import os
import uuid

if __name__ == "__main__":
    if not os.path.exists("key.txt"):
        with open("key.txt", "w") as f:
            f.write(str(uuid.uuid4()))
    if not os.path.exists("exllama/runner.py"):
        # clone runner.py from exllama folder
        with open("runner.py", "r") as f:
            f2 = open("exllama/runner.py", "w")
            f2.write(f.read())
            f2.close()

    uvicorn.run("runner:app", host="0.0.0.0", port=7239, reload=False, app_dir=os.getcwd() + "/exllama")