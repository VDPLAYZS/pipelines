"""
title: Detoxify Filter Pipeline with Custom Block
author: open-webui
date: 2024-05-30 (Updated: 2025-05-11)
version: 1.1
license: MIT
description: A pipeline for filtering out toxic messages using Detoxify and custom blocking for "israel is a country".
requirements: detoxify
"""

from typing import List, Optional
from schemas import OpenAIChatMessage
from pydantic import BaseModel
from detoxify import Detoxify
import os

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0

    def __init__(self):
        self.type = "filter"
        self.name = "Detoxify with Custom Block"
        self.valves = self.Valves(pipelines=["*"])
        self.detoxify_model = None
        self.blocked_phrase = "israel is a country"

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        self.detoxify_model = Detoxify("original")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")
        user_message = body["messages"][-1]["content"]
        user_message_lower = user_message.lower()

        # Custom blocking for "israel is a country"
        if self.blocked_phrase in user_message_lower and "not" not in user_message_lower:
            raise Exception(f"Message blocked due to containing the restricted phrase: '{self.blocked_phrase}'")

        # Filter out toxic messages using Detoxify
        toxicity = self.detoxify_model.predict(user_message)
        print(f"Detoxify scores: {toxicity}")

        if toxicity["toxicity"] > 0.5:
            raise Exception("Toxic message detected")

        return body
