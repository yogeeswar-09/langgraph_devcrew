import os
import logging
from typing import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes

from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dev_crew")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY or GEMINI_API_KEY is missing."
    )


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    max_retries=1,
    timeout=30,
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class DevCrewState(TypedDict, total=False):
    task: str
    plan: str
    implementation: str
    review: str
    output: str


# ============================================================
# PLANNER
# ============================================================

def planner(state: DevCrewState):

    task = state["task"]

    logger.info("DEV CREW: PLANNER")

    return {
        "plan": f"""
Task:

{task}

Development plan:

1. Understand the requirements.
2. Identify the main components.
3. Select appropriate technologies.
4. Design the architecture.
5. Plan implementation.
6. Consider testing, security and deployment.
"""
    }


# ============================================================
# DEVELOPER
# ============================================================

def developer(state: DevCrewState):

    logger.info("DEV CREW: DEVELOPER")

    return {
        "implementation": f"""
Developer analysis for:

{state["task"]}

The solution should include:

- Appropriate project structure
- Required APIs
- Backend/frontend components when applicable
- Database considerations
- Error handling
- Configuration
- Deployment considerations
"""
    }


# ============================================================
# REVIEWER
# ============================================================

def reviewer(state: DevCrewState):

    logger.info("DEV CREW: REVIEWER")

    return {
        "review": f"""
Technical review for:

{state["task"]}

Check:

- Correctness
- Missing requirements
- Security
- Error handling
- Maintainability
- Scalability
- Deployment problems
"""
    }


# ============================================================
# FINALIZER
# ============================================================

def finalizer(state: DevCrewState):

    logger.info("DEV CREW: FINALIZER")

    prompt = f"""
You are Dev Crew, a professional AI software development team.

USER TASK:
{state["task"]}

PLANNER:
{state["plan"]}

DEVELOPER:
{state["implementation"]}

REVIEWER:
{state["review"]}

Produce the final answer.

Use this structure:

# Dev Crew Result

## Understanding

Explain the requested task.

## Architecture

Explain the proposed architecture.

## Implementation

Provide useful implementation details or code.

## Technical Review

Explain important technical considerations.

## How to Run

Explain how to run the solution.

## Next Steps

Give practical next steps.

Be concise but useful.

Do not reveal hidden chain-of-thought.
Do not claim that something was tested unless it was actually tested.
"""

    try:

        response = llm.invoke(prompt)

        content = response.content

        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, dict):

                    parts.append(
                        str(item.get("text", ""))
                    )

                else:

                    parts.append(str(item))

            content = "\n".join(parts)

        return {
            "output": str(content).strip()
        }

    except Exception as error:

        logger.exception("Gemini failed")

        return {
            "output": (
                "# Dev Crew\n\n"
                "The LangGraph workflow completed, "
                "but Gemini returned an error.\n\n"
                f"Error: {error}"
            )
        }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(DevCrewState)

builder.add_node("planner", planner)
builder.add_node("developer", developer)
builder.add_node("reviewer", reviewer)
builder.add_node("finalizer", finalizer)

builder.set_entry_point("planner")

builder.add_edge("planner", "developer")
builder.add_edge("developer", "reviewer")
builder.add_edge("reviewer", "finalizer")
builder.add_edge("finalizer", END)

graph = builder.compile()


# ============================================================
# LANGSERVE FUNCTION
#
# IMPORTANT:
# INPUT IS NOW A PLAIN STRING
# NOT {"input": "..."}
# ============================================================

def run_dev_crew(task: str) -> str:

    logger.info("LANGSERVE REQUEST RECEIVED")
    logger.info("TASK: %s", task)

    if not isinstance(task, str):

        task = str(task)

    task = task.strip()

    if not task:

        return "Please enter a software development task."

    try:

        result = graph.invoke(
            {
                "task": task
            }
        )

        return result.get(
            "output",
            "Dev Crew completed without producing an output."
        )

    except Exception as error:

        logger.exception("LANGGRAPH ERROR")

        return (
            "# Dev Crew Error\n\n"
            f"{error}"
        )


# ============================================================
# RUNNABLE
# ============================================================

agent = RunnableLambda(run_dev_crew)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Dev Crew LangGraph Agent",
    description="Multi-stage software development agent built with LangGraph",
    version="1.0.0",
)


# ============================================================
# LANGSERVE
#
# IMPORTANT:
# input_type=str
# output_type=str
#
# This removes the required 'input' object problem.
# ============================================================

add_routes(
    app,
    agent,
    path="/agent",
    input_type=str,
    output_type=str,
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "Dev Crew",
        "framework": "LangGraph",
        "status": "running",
        "workflow": [
            "Planner",
            "Developer",
            "Reviewer",
            "Finalizer"
        ],
        "playground": "/agent/playground/"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "agent": "Dev Crew",
        "framework": "LangGraph"
    }


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            "8000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
