from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import hu_core_news_trf
import requests
import random
import click
import os
import json
import itertools


def merge_disjointed_names(ner_results: list):
    names_to_change = []
    name_positions = []
    previous = {}
    for result in ner_results:
        if result.get("start") == previous.get("end"):
            names_to_change.pop(-1)
            name_positions.pop(-1)
            names_to_change.append(
                (previous.get("word") + result.get("word")).replace("#", "")
            )
            name_positions.append(
                {"start": previous.get("start"), "end": result.get("end")}
            )
        else:
            names_to_change.append(result.get("word"))
            name_positions.append(
                {"start": result.get("start"), "end": result.get("end")}
            )
        previous = result

    return names_to_change, name_positions


def recognise_people(input: str):
    tokenizer = AutoTokenizer.from_pretrained(
        "NYTK/named-entity-recognition-nerkor-hubert-hungarian"
    )
    model = AutoModelForTokenClassification.from_pretrained(
        "NYTK/named-entity-recognition-nerkor-hubert-hungarian"
    )

    ner = pipeline("ner", model=model, tokenizer=tokenizer)
    temp_results = ner(input)
    ner_results = []
    for result in temp_results:
        if "PER" in result.get("entity"):
            ner_results.append(result)

    return merge_disjointed_names(ner_results)


def paginate_ner(text: str):
    parts = [text[i : i + 500] for i in range(0, len(text), 400)] #FIXME split at space
    results = [recognise_people(part) for part in parts]
    r_names, r_positions = [],[]
    for names, positions in results:
        r_names.append(names)
        r_positions.append(positions)
    return r_names, r_positions


def morphological_analysis_husplacy(names_to_change: list[str]):
    nlp = hu_core_news_trf.load()
    name_lemmas = []
    name_morphs = []
    for name in names_to_change:
        doc = nlp(name)
        name_lemma = ""
    for token in doc:
        name_lemmas.append(token.lemma_)
        name_morphs.append(token.morph.to_json())

    return name_lemmas, name_morphs


def _send_emagyar_request(text: str):
    r = requests.post("http://127.0.0.1:5000/tok/morph", data={"text": "Ilonával"})
    resp = r.text.split("\t")[-1]
    info = json.loads(resp)
    name_lemma = info[0]["lemma"]
    name_morph = info[0]["tag"]
    return name_lemma, name_morph


def morphological_analysis_emagyar(names_to_change: list[str]):
    name_lemmas = []
    name_morphs = []
    for name in names_to_change:
        name_lemma, name_morph = _send_emagyar_request(name)
        name_lemmas.append(name_lemma)
        name_morphs.append(name_morph)
    return name_lemmas, name_morphs


def find_pseudonyms_for_lemmas(name_lemmas: list[str]):
    female_names = []
    male_names = []

    with open("/content/female_names.txt", "r") as f:
        for line in f:
            female_names.append(line.strip())
    with open("/content/male_names.txt", "r") as f:
        for line in f:
            male_names.append(line.strip())
    name_pseudonyms = []
    for name in name_lemmas:
        if name in male_names:
            name_pseudonyms.append(random.choice(male_names))
        elif name in female_names:
            name_pseudonyms.append(random.choice(female_names))
        else:
            name_pseudonyms.append(name)
    return name_pseudonyms


def _generate_word_form(word_with_tag: str):
    body = {"text": word_with_tag}
    response = requests.post("http://dl3.nytud.hu:60004/translate", data=body)
    print(response.json())


def run_emagyar_pipeline(text: str):
    people_names, name_positions = paginate_ner(text)
    name_lemmas, name_morphs = morphological_analysis_emagyar(people_names)
    peudonyms = find_pseudonyms_for_lemmas(name_lemmas)
    


def run_huspacy_pipeline(text: str):
    people_names, name_positions = paginate_ner(text)
    name_lemmas, name_morphs = morphological_analysis_husplacy(people_names)
    peudonyms = find_pseudonyms_for_lemmas(name_lemmas)


@click.command()
@click.option("--file-input", help="path of the input file")
@click.option("--format", help="pipeline to run: emagyar, huspacy")
@click.option("--only-ner", help="only run the NER on the input")
def process(file_input, format, only_ner):
    with open(os.path.join(os.getcwd(), file_input), "r", encoding="utf8") as f:
        text = f.readlines()
        text = "".join(text)
    if only_ner:
        print(paginate_ner(text))
    if format == "emagyar":
        run_emagyar_pipeline(text)
    else:
        run_huspacy_pipeline(text)


if __name__ == "__main__":
    process()
