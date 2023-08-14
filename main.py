import uvicorn
import os

if __name__ == "__main__":
    if not os.path.exists("exllama/runner.py"):
        # clone runner.py from exllama folder
        with open("runner.py", "r") as f:
            f2 = open("exllama/runner.py", "w")
            f2.write(f.read())
            f2.close()

    uvicorn.run("runner:app", host="0.0.0.0", port=7239, reload=False, app_dir=os.getcwd() + "/exllama")