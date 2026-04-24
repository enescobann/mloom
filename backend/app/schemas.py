from typing import Any
from pydantic import BaseModel, ConfigDict
import datetime
from enum import Enum

class RunTypeEnum(str, Enum):
    LLM = "LLM"
    CLASSIC = "CLASSIC"


class LLMMetricsBase(BaseModel):
    model_name: str | None = None
    prompt: str | None = None
    response: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_cost: float | None = None
    latency: int | None = None

class LLMMetricsCreate(LLMMetricsBase):
    pass
class LLMMetricsResponse(LLMMetricsBase):
    id: int
    run_id: int

    model_config = ConfigDict(from_attributes=True)


class RunBase(BaseModel):
    run_name: str
    run_type: RunTypeEnum
    tags: dict[str, Any] | None = None
    latency: int | None = None

class RunCreate(RunBase):
    project_id: int
    metrics: list[LLMMetricsCreate] = []

class RunResponse(RunBase):
    id: int
    project_id: int
    timestamp: datetime.datetime
    metrics: LLMMetricsResponse = []

    model_config = ConfigDict(from_attributes=True)


class ProjectBase(BaseModel):
    name: str

class ProjectCreate(ProjectBase):
    pass

class ProjectResponse(ProjectBase):
    id: int
    created_at: datetime.datetime
    runs: list[RunResponse] = []

    model_config = ConfigDict(from_attributes=True)