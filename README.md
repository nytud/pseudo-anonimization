# Pseudo anonimization

## Usage

---

- cli
- server

---

## Installation for CLI

1. clone repository
2. create virtual environment

```bash
windows: 
python3 -m venv .ve
unix like:
python3 -m venv .ve
```

3. activate environment / resource termnal

```bash
windows:
& .ve/Scripts/Acitvate.ps1
unix like:
source .ve/bin/activate
```

4. install requirements

```bash
pip install -r requirements.txt
```

### Starting

start docker container for emtsv

```bash
docker run --rm -p5000:5000 -it mtaril/emtsv 
```

start application - result will be written to stdout

```bash
python .\anonimization.py --file-input "path/to/file" --format=[emagyar, huspacy]
```

## Starting as server

```bash
docker compose up -d
```

the server is available on port 8000
available endpoints:

- /tokenize/emagyar : only tokenizes the input
- /tokenize/huspacy
- /swap/emagyar
- /swap/huspacy

all endpoints requires a file input or body:{"text":"text to  process"}
