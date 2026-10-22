"""
Minimal serving process for detector.py.

Run:
    uvicorn serve:app --host 0.0.0.0 --port 8001

Then, e.g. from Java (or curl):
    POST http://localhost:8001/spikes
    {"date": "2026-10-04",
     "students": [{"id": "1234567", "counts": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,11,8,2,2,0,0,0,0,0]}]}
    -> {"date": "2026-10-04", "results": [{"id": "1234567", "timestamps": ["14:00"]}]}

Only students with at least one spike appear in "results".
"""

from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, Field

from detector import HOURS_PER_DAY, detect_spikes

app = FastAPI()


class StudentCounts(BaseModel):
    id: str
    counts: list[Annotated[int, Field(ge=0)]] = Field(min_length=HOURS_PER_DAY, max_length=HOURS_PER_DAY)


class SpikeRequest(BaseModel):
    date: str
    students: list[StudentCounts]


class StudentSpikes(BaseModel):
    id: str
    timestamps: list[str]


class SpikeResponse(BaseModel):
    date: str
    results: list[StudentSpikes]


@app.post("/spikes", response_model=SpikeResponse)
def spikes(req: SpikeRequest):
    results = []
    for student in req.students:
        hours = detect_spikes(student.counts)
        if hours:
            results.append(StudentSpikes(id=student.id, timestamps=hours))
    return SpikeResponse(date=req.date, results=results)


@app.get("/health")
def health():
    return {"status": "ok"}
