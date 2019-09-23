# Datadriven Machine Learning

Machine learning project for predictive maintenance of industrial components. In order to run things correctly, remember to follow the steps in both the _Predictive Maintenance_ part and the _Web Application part_.

## Predictive Maintenance

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

6. Install packages using requirements_local.txt
   pip install -r requirements_local.txt

## Web application

This project contains a React web application and a server for uploading files. This project assumes you have a Node package manager installed (Yarn or NPM).

Inside the app folder, install all dependencies:

```shell
$ yarn
```

or

```shell
$ npm install
```

Then start the localhost:

```shell
$ yarn start
```

or

```shell
$ npm start
```

To install the dependencies for the server, navigate to the server folder. Install the required dependencies and start the server:

```shell
$ npm install
$ node server.js
```
