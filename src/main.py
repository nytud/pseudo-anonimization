import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from anonimization import paginate_ner, run_emagyar_pipeline, run_huspacy_pipeline, process

logging.basicConfig(level=logging.DEBUG)
app = FastAPI()


class Text(BaseModel):
    text: str


@app.get("/ping")
def read_root():
    return {"ping": "pong"}


@app.post("/anonymization")
async def anonymization(item: Text):
    result = process(item.text, morph_code_type="emagyar", only_ner=False, is_consistent=False)
    return result


@app.post("/tokenize/emagyar")
async def emagyar_only_tok(
        file: Optional[UploadFile] = None, text: Optional[Text] = Body(embed=True)
):
    if not file and not text:
        raise HTTPException(
            status_code=422,
            detail="No file or text provided in the body of the request",
        )
    if file:
        contents = str(await file.read(), "utf8")
        return paginate_ner(contents, True)
    if text:
        return paginate_ner(text, True)


@app.post("/tokenize/huspacy")
async def huspacy_only_tok(
        file: Optional[UploadFile] = None, text: Optional[Text] = Body(embed=True)
):
    if not file and not text:
        raise HTTPException(
            status_code=422,
            detail="No file or text provided in the body of the request",
        )
    if file:
        contents = str(await file.read(), "utf8")
        return paginate_ner(contents, False)
    if text:
        return paginate_ner(text, False)


@app.post("/swap/emagyar")
async def emagyar_full_pipeline(
        file: Optional[UploadFile] = None, text: Optional[Text] = Body(embed=True)
):
    if not file and not text:
        raise HTTPException(
            status_code=422,
            detail="No file or text provided in the body of the request",
        )
    if file:
        contents = str(await file.read(), "utf8")
        return run_emagyar_pipeline(contents)
    if text:
        return run_emagyar_pipeline(text)


@app.post("/swap/huspacy")
async def husplacy_full_pipeline(
        file: Optional[UploadFile] = None, text: Optional[Text] = Body(embed=True)
):
    if not file and not text:
        raise HTTPException(
            status_code=422,
            detail="No file or text provided in the body of the request",
        )
    if file:
        contents = str(await file.read(), "utf8")
        return run_huspacy_pipeline(contents)
    if text:
        return run_huspacy_pipeline(text)
