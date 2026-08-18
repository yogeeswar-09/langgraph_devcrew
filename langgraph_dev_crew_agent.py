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
    max_retries=1,
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
# PLAYGROUND SCHEMAS
# ============================================================

class DevCrewInput(BaseModel):
    input: str = Field(
        default="",
        description="Describe the software development task."
    )


class DevCrewOutput(BaseModel):
    output: str


# ============================================================
# NODE 1 — PLANNER
# ============================================================

def planner_node(state: DevCrewState) -> DevCrewState:
    request = state.get("request", "").strip()

    # Keep this stage fast and deterministic.
    plan = f"""Development Plan

Goal:
Solve the following development request:

{request}

Planning checklist:
1. Understand the requested functionality.
2. Identify the required frontend/backend/data components.
3. Choose a practical technology approach.
4. Consider validation, errors, security, and deployment.
5. Produce an implementation that is simple enough to maintain.
"""

    logger.info("Planner completed")
    return {"plan": plan}


# ============================================================
# NODE 2 — DEVELOPER
# ============================================================

def developer_node(state: DevCrewState) -> DevCrewState:
    request = state.get("request", "")
    plan = state.get("plan", "")

    # This stage prepares the implementation task for the final
    # Gemini call instead of making another expensive API request.
    implementation = f"""Developer Task

Original request:
{request}

Plan:
{plan}

The final developer response should provide:
- A practical architecture
- Complete relevant code where appropriate
- Clear setup instructions
- Error handling
- Maintainable implementation
"""

    logger.info("Developer stage completed")
    return {"implementation": implementation}


# ============================================================
# NODE 3 — REVIEWER
# ============================================================

def reviewer_node(state: DevCrewState) -> DevCrewState:
    request = state.get("request", "")
    implementation = state.get("implementation", "")

    review = f"""Senior Review Checklist

Request:
{request}

Review the proposed solution for:
- Correctness
- Missing requirements
- Security
- Error handling
- Maintainability
- Deployment/runtime concerns

The final response must avoid claiming that code was tested unless it
was actually executed.
"""

    logger.info("Reviewer stage completed")
    return {"review": review}


# ============================================================
# NODE 4 — FINALIZER
# ============================================================

def finalizer_node(state: DevCrewState) -> DevCrewState:
    request = state.get("request", "")
    plan = state.get("plan", "")
    implementation = state.get("implementation", "")
    review = state.get("review", "")

    prompt = f"""
You are Dev Crew, a professional software development team.

You have four internal stages:
Planner -> Developer -> Reviewer -> Finalizer.

Create the final answer for this user request.

USER REQUEST:
{request}

PLANNER:
{plan}

DEVELOPER:
{implementation}

REVIEWER:
{review}

Return a useful, human-readable answer using this structure:

## Dev Crew Result

### Approach
Explain the recommended solution briefly.

### Implementation
Provide complete relevant code when code is requested.
Use markdown code blocks.

### Review
Mention important correctness, security, and maintainability considerations.

### Next Steps
Give concise instructions to run, test, or deploy the solution.

Important:
- Do not reveal chain-of-thought.
- Do not claim that code was executed or tested.
- Do not invent files, APIs, or test results.
- Keep the answer practical and developer-ready.
"""

    logger.info("Finalizer calling Gemini")

    try:
        response = llm.invoke(prompt)
        answer = response.content

        if isinstance(answer, list):
            answer = "\n".join(
                str(item.get("text", item))
                if isinstance(item, dict)
                else str(item)
                for item in answer
            )

        return {"final_answer": str(answer)}

    except Exception as exc:
        logger.exception("Gemini finalizer failed")
        return {
            "final_answer": (
                "Dev Crew could not complete the Gemini response.\n\n"
                f"Error: {exc}"
            )
        }


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

    logger.info("Dev Crew request received: %s", request)

    try:
        result = dev_crew_graph.invoke(
            {"request": request}
        )

        return DevCrewOutput(
            output=result.get(
                "final_answer",
                "Dev Crew completed without a final response."
            )
        )

    except Exception as exc:
        logger.exception("Dev Crew graph failed")

        return DevCrewOutput(
            output=(
                "Dev Crew encountered an error.\n\n"
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
    version="2.0.0",
    description=(
        "Fast LangGraph development workflow: "
        "Planner -> Developer -> Reviewer -> Finalizer."
    ),
)

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
# RUN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "langgraph_dev_crew_agent:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
