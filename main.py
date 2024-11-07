from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
from mrz import ExtractMRZ


class Image_url(BaseModel):
    url: str


@app.post("/api/mrz")
def read_item(url: Image_url):
    data = ExtractMRZ(url.url)
    return data.finish()
