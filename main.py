from http.client import HTTPResponse
from typing import Optional
from pydantic import BaseModel

from fastapi import FastAPI, UploadFile, status

from anonimization import paginate_ner, run_emagyar_pipeline, run_huspacy_pipeline

app = FastAPI()


class Text(BaseModel):
    text: str


@app.get("/ping")
def read_root():
    return {"ping": "pong"}


@app.get("/tokenize/emagyar")
def emagyar_only_tok(file: Optional[UploadFile], text: Optional[Text]):
    if not file and not text:
        return HTTPResponse(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="No file or text provided in the body of the request",
        )
    if file:
        contents = str(file.read())
        return paginate_ner(contents, True)
    if text:
        return paginate_ner(text, True)


@app.get("/tokenize/huspacy")
def huspacy_only_tok(file: Optional[UploadFile], text: Optional[Text]):
    if not file and not text:
        return HTTPResponse(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="No file or text provided in the body of the request",
        )
    if file:
        contents = str(file.read())
        return paginate_ner(contents, False)
    if text:
        return paginate_ner(text, False)


@app.get("/swap/emagyar")
def emagyar_full_pipeline(file: Optional[UploadFile], text: Optional[Text]):
    if not file and not text:
        return HTTPResponse(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="No file or text provided in the body of the request",
        )
    if file:
        contents = str(file.read())
        return run_emagyar_pipeline(contents)
    if text:
        return run_emagyar_pipeline(text)


@app.get("/swap/huspacy")
def husplacy_full_pipeline(file: Optional[UploadFile], text: Optional[Text]):
    if not file and not text:
        return HTTPResponse(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="No file or text provided in the body of the request",
        )
    if file:
        contents = str(file.read())
        return run_huspacy_pipeline(contents)
    if text:
        return run_huspacy_pipeline(text)
