import os
import logging
from typing import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY or GEMINI_API_KEY is missing. "
        "Add it in Render Environment Variables."
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dev_crew")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


class DevCrewState(TypedDict, total=False):
    request: str
    plan: str
    implementation: str
    review: str
    final_answer: str


def planner_node(state: DevCrewState) -> DevCrewState:
    request = state.get("request", "").strip()

    if not request:
        return {"plan": "No development request was provided."}

    response = llm.invoke(f"""
You are the Planner of Dev Crew, a professional software development team.

Analyze the user's development request and create a practical implementation plan.

User request:
{request}

Return:
1. Goal
2. Requirements
3. Technical approach
4. Files/components likely needed
5. Step-by-step implementation plan
6. Important edge cases

Be concise but useful. Do not write the full implementation yet.
""")

    return {"plan": response.content}


def developer_node(state: DevCrewState) -> DevCrewState:
    request = state.get("request", "")
    plan = state.get("plan", "")

    response = llm.invoke(f"""
You are the Developer of Dev Crew.

Implement the user's request using the plan created by the Planner.

USER REQUEST:
{request}

PLANNER'S PLAN:
{plan}

Produce a practical implementation.

Rules:
- Prefer simple, maintainable solutions.
- Use clear code.
- Explain important implementation decisions.
- If code is required, provide complete relevant code rather than pseudocode.
- Do not claim that code was executed or tested when it was not.
""")

    return {"implementation": response.content}


def reviewer_node(state: DevCrewState) -> DevCrewState:
    request = state.get("request", "")
    implementation = state.get("implementation", "")

    response = llm.invoke(f"""
You are the Senior Reviewer of Dev Crew.

Review the proposed solution below against the original request.

ORIGINAL REQUEST:
{request}

PROPOSED IMPLEMENTATION:
{implementation}

Check for:
- Correctness
- Missing requirements
- Bugs
- Security issues
- Poor assumptions
- Maintainability
- Deployment/runtime concerns

Return:
VERDICT: PASS or NEEDS CHANGES

Then give a concise review and exact fixes if needed.
""")

    return {"review": response.content}


def finalizer_node(state: DevCrewState) -> DevCrewState:
    request = state.get("request", "")
    plan = state.get("plan", "")
    implementation = state.get("implementation", "")
    review = state.get("review", "")

    response = llm.invoke(f"""
You are the Finalizer of Dev Crew.

Create the final response for the developer.

ORIGINAL REQUEST:
{request}

PLAN:
{plan}

IMPLEMENTATION:
{implementation}

REVIEW:
{review}

Return a polished developer-ready answer.

Structure it as:

## Dev Crew Result

### Approach
Briefly explain the approach.

### Implementation
Give the final implementation or the most useful complete code.

### Review
Summarize the review and whether the solution is ready.

### Next Steps
Give practical steps to run, test, or deploy it.

Do not mention internal chain-of-thought.
Do not invent test results.
""")

    return {"final_answer": response.content}


graph_builder = StateGraph(DevCrewState)

graph_builder.add_node("planner", planner_node)
graph_builder.add_node("developer", developer_node)
graph_builder.add_node("reviewer", reviewer_node)
graph_builder.add_node("finalizer", finalizer_node)

graph_builder.set_entry_point("planner")
graph_builder.add_edge("planner", "developer")
graph_builder.add_edge("developer", "reviewer")
graph_builder.add_edge("reviewer", "finalizer")
graph_builder.add_edge("finalizer", END)

dev_crew_graph = graph_builder.compile()


def run_dev_crew(payload):
    if isinstance(payload, str):
        request = payload.strip()
    elif isinstance(payload, dict):
        request = str(payload.get("input", "")).strip()
    else:
        request = ""

    if not request:
        return {"error": "Please enter a development request."}

    logger.info("Dev Crew request received: %s", request)

    try:
        result = dev_crew_graph.invoke({"request": request})

        return {
            "input": request,
            "plan": result.get("plan", ""),
            "implementation": result.get("implementation", ""),
            "review": result.get("review", ""),
            "final_answer": result.get("final_answer", ""),
        }

    except Exception as exc:
        logger.exception("Dev Crew execution failed")
        return {
            "input": request,
            "error": "Dev Crew could not complete the request.",
            "details": str(exc),
        }


dev_crew_runnable = RunnableLambda(run_dev_crew)

app = FastAPI(
    title="Dev Crew - LangGraph Agent",
    version="1.0.0",
    description=(
        "LangGraph development workflow: "
        "Planner -> Developer -> Reviewer -> Finalizer."
    ),
)

add_routes(app, dev_crew_runnable, path="/agent")


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
    return {"status": "healthy", "agent": "Dev Crew"}


if __name__ == "__main__":
    uvicorn.run(
        "langgraph_dev_crew_agent:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
