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
pip install  https://huggingface.co/huspacy/hu_core_news_trf/resolve/main/hu_core_news_trf-any-py3-none-any.whl
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

component diagram 
@startuml
agent text
queue "morphological analysis" as morpho
database "Hungarian given names" as given
queue "generate form of pseudo anonymized name" as gen
queue NER

component emtsv
component huspacy
component NerKor
component PseudoAnonimizator as pseu

text --> pseu
pseu -right-> NER
NER --> NerKor
NerKor --> pseu
pseu -right-> morpho
morpho -- emtsv
morpho -- huspacy
pseu -right-> given : select pseudo name
pseu --> gen
gen -- emtsv
gen -- huspacy

@enduml

