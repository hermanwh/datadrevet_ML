# datadrevet_ML
Machine learning project for predictive maintenance of industrial components

1. Install Python 3.6 (no need to put in path)

2. Update pip
python -m pip install --upgrade pip

3. Install virtualenv
python -m pip install --user virtualenv 

4. Create venv with Python 3.6
python -m virtualenv -p path\to\python\3.6\python.exe venv
e.g.
python -m virtualenv -p C:\Users\Herman\AppData\Local\Programs\Python\Python36\python.exe venv

5. Activate virtual environment
venv\scripts\activate

5. Install packages using requirements_local.txt
pip install -r requirements_local.txt