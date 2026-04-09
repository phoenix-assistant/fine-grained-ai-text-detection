"""MCP server for AI text detection."""

from __future__ import annotations


def create_app():
    from fastapi import FastAPI
    from pydantic import BaseModel
    from aidetect.detector import Detector

    app = FastAPI(title="aidetect MCP", version="0.1.0")
    detector = Detector()

    class AnalyzeRequest(BaseModel):
        text: str

    @app.post("/analyze")
    def analyze(req: AnalyzeRequest):
        result = detector.analyze(req.text)
        return result.to_dict()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
