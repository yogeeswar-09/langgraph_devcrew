import os
import logging
from typing import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY or GEMINI_API_KEY is missing. "
        "Add it to Render Environment Variables."
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dev_crew")


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class DevCrewState(TypedDict, total=False):
    request: str
    plan: str
    implementation: str
    review: str
    final_answer: str


# ============================================================
# PLAYGROUND INPUT SCHEMA
# IMPORTANT:
# LangServe needs a concrete input schema to render the
# /agent/playground/ input box.
# ============================================================

class DevCrewInput(BaseModel):
    input: str = Field(
        default="",
        description="Describe the software development task you want Dev Crew to solve."
    )


class DevCrewOutput(BaseModel):
    output: str


# ============================================================
# GRAPH NODES
# ============================================================

def planner_node(state: DevCrewState) -> DevCrewState:
    request = state.get("request", "").strip()

    response = llm.invoke(
        f"""
You are the Planner in a software development team called Dev Crew.

Analyze this development request:

{request}

Create a practical plan containing:
1. Goal
2. Requirements
3. Technical approach
4. Main components/files
5. Implementation steps
6. Important edge cases

Do not write the full implementation yet.
Keep the plan concise and practical.
"""
    )

    return {"plan": response.content}


def developer_node(state: DevCrewState) -> DevCrewState:
    request = state.get("request", "")
    plan = state.get("plan", "")

    response = llm.invoke(
        f"""
You are the Developer in Dev Crew.

Original request:
{request}

Planner's plan:
{plan}

Now produce the implementation.

Rules:
- Prefer simple, maintainable solutions.
- If code is required, provide complete relevant code.
- Do not use pseudocode when real code can be provided.
- Explain important implementation decisions.
- Never claim that code was executed or tested unless it actually was.
"""
    )

    return {"implementation": response.content}


def reviewer_node(state: DevCrewState) -> DevCrewState:
    request = state.get("request", "")
    implementation = state.get("implementation", "")

    response = llm.invoke(
        f"""
You are the Senior Code Reviewer in Dev Crew.

Original request:
{request}

Proposed implementation:
{implementation}

Review it for:
- Correctness
- Missing requirements
- Bugs
- Security issues
- Maintainability
- Runtime/deployment problems

Start with exactly one of:

VERDICT: PASS

or

VERDICT: NEEDS CHANGES

Then give a concise review and specific fixes where necessary.
"""
    )

    return {"review": response.content}


def finalizer_node(state: DevCrewState) -> DevCrewState:
    request = state.get("request", "")
    plan = state.get("plan", "")
    implementation = state.get("implementation", "")
    review = state.get("review", "")

    response = llm.invoke(
        f"""
You are the Finalizer of Dev Crew.

Create the final developer-ready response.

Original request:
{request}

Plan:
{plan}

Implementation:
{implementation}

Review:
{review}

Use this structure:

## Dev Crew Result

### Approach
Brief explanation.

### Implementation
Provide the final useful implementation/code.

### Review
Summarize the review and whether the solution is ready.

### Next Steps
Give practical run, test, or deployment steps.

Do not reveal chain-of-thought.
Do not invent test results.
"""
    )

    return {"final_answer": response.content}


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(DevCrewState)

builder.add_node("planner", planner_node)
builder.add_node("developer", developer_node)
builder.add_node("reviewer", reviewer_node)
builder.add_node("finalizer", finalizer_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "developer")
builder.add_edge("developer", "reviewer")
builder.add_edge("reviewer", "finalizer")
builder.add_edge("finalizer", END)

dev_crew_graph = builder.compile()


# ============================================================
# LANGSERVE ADAPTER
# ============================================================

def run_dev_crew(data: DevCrewInput) -> DevCrewOutput:
    request = (data.input or "").strip()

    if not request:
        return DevCrewOutput(
            output="Enter a development request above, then click Start."
        )

    logger.info("Dev Crew request: %s", request)

    try:
        result = dev_crew_graph.invoke(
            {
                "request": request
            }
        )

        # Return a single text field so the LangServe Playground
        # has a normal text renderer for the output.
        return DevCrewOutput(
            output=result.get(
                "final_answer",
                "Dev Crew completed without a final response."
            )
        )

    except Exception as exc:
        logger.exception("Dev Crew execution failed")

        return DevCrewOutput(
            output=(
                "Dev Crew encountered an error while processing "
                "your request.\n\n"
                f"Error: {exc}"
            )
        )


dev_crew_runnable = RunnableLambda(run_dev_crew).with_types(
    input_type=DevCrewInput,
    output_type=DevCrewOutput,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Dev Crew - LangGraph Agent",
    version="1.1.0",
    description=(
        "LangGraph development workflow: "
        "Planner -> Developer -> Reviewer -> Finalizer."
    ),
)


# This creates:
# /agent
# /agent/invoke
# /agent/batch
# /agent/stream
# /agent/playground/
add_routes(
    app,
    dev_crew_runnable,
    path="/agent",
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "agent": "Dev Crew",
        "framework": "LangGraph",
        "status": "running",
        "workflow": "Planner -> Developer -> Reviewer -> Finalizer",
        "playground": "/agent/playground/",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "agent": "Dev Crew",
    }


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "langgraph_dev_crew_agent:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
