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
    start_time: datetime.datetime | None = None
    end_time: datetime.datetime | None = None

class LLMMetricsCreate(LLMMetricsBase):
    pass
class LLMMetricsResponse(LLMMetricsBase):
    id: int
    run_id: int
    timestamp: datetime.datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class RunBase(BaseModel):
    run_name: str
    run_type: RunTypeEnum = RunTypeEnum.LLM
    tags: dict[str, Any] | None = None
    latency: int | None = None
    error_trace: dict | None = None
    call_site: dict | None = None
    start_time: datetime.datetime | None = None
    end_time: datetime.datetime | None = None

class RunCreate(RunBase):
    # Older clients may send numeric ids; ids are strings everywhere.
    model_config = ConfigDict(coerce_numbers_to_str=True)

    project_id: str
    metrics: list[LLMMetricsCreate] = []

class RunResponse(RunBase):
    id: int
    project_id: str
    timestamp: datetime.datetime | None = None
    metrics: list[LLMMetricsResponse] = []

    model_config = ConfigDict(from_attributes=True)


class ProjectBase(BaseModel):
    name: str

class ProjectCreate(ProjectBase):
    # Derived from the name when omitted.
    id: str | None = None

class ProjectUpdate(ProjectBase):
    pass

class ProjectResponse(ProjectBase):
    id: str
    created_at: datetime.datetime | None = None

    model_config = ConfigDict(from_attributes=True)

class ProjectSummary(ProjectResponse):
    run_count: int = 0
