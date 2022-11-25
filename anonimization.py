import csv
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import hu_core_news_trf
import requests
import random
import click
import os
import json
import torch
from os import environ


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
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(
        "NYTK/named-entity-recognition-nerkor-hubert-hungarian"
    )
    model = AutoModelForTokenClassification.from_pretrained(
        "NYTK/named-entity-recognition-nerkor-hubert-hungarian"
    )

    ner = pipeline("ner", model=model, tokenizer=tokenizer, device=device)
    temp_results = ner(input)
    ner_results = []
    for result in temp_results:
        if "PER" in result.get("entity"):
            ner_results.append(result)

    return merge_disjointed_names(ner_results)


def tokenize_emagyar(text: str):
    r = requests.post(f"http://{environ['host']}:5000/tok", data={"text": text})
    sentences = []
    current_sentence = ""  # FIXME read tsv
    for line in r.text.split("\n")[1:]:
        if not line:
            sentences.append(current_sentence + "\n")
            current_sentence = ""
            continue
        line = line.replace('"', "")
        line = line.replace("\\n", "\n")
        word = line.split("\t")[0]
        current_sentence += word + line.split("\t")[1]
    return sentences


def tokenize_huspacy(text: str) -> list[str]:
    nlp = hu_core_news_trf.load()
    segments = nlp(text)
    sentences = []
    current = ""
    for token in segments:
        if token == "\n":
            sentences.append(current)
            current = ""
            continue
        current += token
    return sentences


def paginate_ner(text: str, is_emagyar: bool = False):
    if is_emagyar:
        sentences = tokenize_emagyar(text)
    else:
        sentences = tokenize_huspacy(text)
    results = [(recognise_people(part), part) for part in sentences]
    print(results)
    return results


def morphological_analysis_huspacy(names_to_change: list[str]):
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
    r = requests.post(f"http://{environ['host']}:5000/tok/morph", data={"text": text})
    resp = r.text.split("\t")[-1]
    info = json.loads(resp)
    if not info:
        return "", ""
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


def find_pseudonyms_for_lemmas(name_lemmas: list[str], is_consistent:bool = True):
    female_names = set()
    male_names = set()
    used_pseudo_names = {}

    with open("./content/female_names.txt", "r", encoding="utf8") as f:
        for line in f:
            female_names.add(line.strip())
    with open("./content/male_names.txt", "r", encoding="utf8") as f:
        for line in f:
            male_names.add(line.strip())
    name_pseudonyms = []
    for name in name_lemmas:
        if name in used_pseudo_names and not is_consistent:
            name_pseudonyms.append(used_pseudo_names[name])
        elif name in male_names:
            chosen_pseudo_name = random.choice(male_names)
            used_pseudo_names[name] = chosen_pseudo_name
            name_pseudonyms.append(chosen_pseudo_name)
        elif name in female_names:
            chosen_pseudo_name = random.choice(female_names)
            used_pseudo_names[name] = chosen_pseudo_name
            name_pseudonyms.append(chosen_pseudo_name)
        else:
            name_pseudonyms.append(name)
    return name_pseudonyms


def _generate_word_form(word_with_tag: str, is_emagyar: bool = True):
    url = "https://juniper.nytud.hu/demo/nlp/trans/morph-ud"
    if is_emagyar:
        url = "https://juniper.nytud.hu/demo/nlp/trans/morph-em"
    payload = json.dumps({"text": word_with_tag})
    response = requests.request(
        "POST", url, headers={"Content-Type": "application/json"}, data=payload
    )
    return response.json()["text"]


def run_emagyar_pipeline(text: str, is_consistent:bool):
    zipped = paginate_ner(text, True)
    result = []
    for elem in zipped:
        double, sentence = elem
        people_names, name_positions = double
        name_lemmas, name_morphs = morphological_analysis_emagyar(people_names)
        pseudonyms = find_pseudonyms_for_lemmas(name_lemmas,is_consistent )
        delta = 0
        for position, morph, pseudonym in zip(name_positions, name_morphs, pseudonyms):
            name_with_tag = pseudonym + morph
            generated = _generate_word_form(name_with_tag)
            original_length = position["end"] - position["start"]
            new_length = len(generated)
            sentence = (
                sentence[: position["start"] - delta] + generated + sentence[position["end"] - delta :]
            )
            delta = original_length - new_length
        result.append(sentence)
    print(result)
    return result


def run_huspacy_pipeline(text: str, is_consistent:bool):
    zipped = paginate_ner(text, False)
    result = []
    for elem in zipped:
        double, sentence = elem
        people_names, name_positions = double
        name_lemmas, name_morphs = morphological_analysis_huspacy(people_names)
        pseudonyms = find_pseudonyms_for_lemmas(name_lemmas. is_consistent)
        for position, morph, pseudonym in zip(name_positions, name_morphs, pseudonyms):
            name_with_tag = pseudonym + morph
            generated = _generate_word_form(name_with_tag, False)
            sentence = (
                sentence[: position["start"]] + generated + sentence[position["end"] :]
            )
        result.append(sentence)
    print(result)
    return result


@click.command()
@click.option("--file-input", help="path of the input file")
@click.option("--format", help="pipeline to run: emagyar, huspacy")
@click.option("--only-ner", help="only run the NER on the input")
@click.option("--is-consistent", help="the same name will be changed consistently in the text")
def process(file_input, format, only_ner, is_consistent=True):
    environ["host"] = "localhost"
    with open(os.path.join(os.getcwd(), file_input), "r", encoding="utf8") as f:
        text = f.readlines()
        text = "".join(text)
    if only_ner and format == "emagyar":
        paginate_ner(text, True)
        return
    if only_ner:
        paginate_ner(text)
        return
    if format == "emagyar":
        run_emagyar_pipeline(text, is_consistent)
    else:
        run_huspacy_pipeline(text, is_consistent)


if __name__ == "__main__":
    process()
