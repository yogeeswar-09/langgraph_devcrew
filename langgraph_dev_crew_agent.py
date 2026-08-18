import os
import logging
from typing import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dev_crew")

API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if not API_KEY:
    raise RuntimeError(
        "Missing GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
    )


# ============================================================
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=API_KEY,
    temperature=0,
    max_retries=0,
    timeout=25,
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class DevCrewState(TypedDict, total=False):
    input: str
    plan: str
    implementation: str
    review: str
    output: str


# ============================================================
# NODE 1 — PLANNER
# ============================================================

def planner(state: DevCrewState):

    user_input = state.get("input", "").strip()

    logger.info("NODE 1: Planner")

    plan = f"""
DEVELOPMENT PLAN

User request:
{user_input}

Plan:
1. Understand the requested application.
2. Identify the main components.
3. Select appropriate technologies.
4. Design the implementation.
5. Consider validation, security and deployment.
"""

    return {
        "plan": plan
    }


# ============================================================
# NODE 2 — DEVELOPER
# ============================================================

def developer(state: DevCrewState):

    logger.info("NODE 2: Developer")

    implementation = f"""
DEVELOPER ANALYSIS

User request:
{state.get("input", "")}

Planning information:
{state.get("plan", "")}

The implementation should be:
- Practical
- Maintainable
- Production-oriented
- Easy for a student developer to understand
"""

    return {
        "implementation": implementation
    }


# ============================================================
# NODE 3 — REVIEWER
# ============================================================

def reviewer(state: DevCrewState):

    logger.info("NODE 3: Reviewer")

    review = f"""
CODE REVIEW

Check the proposed solution for:

- Correctness
- Missing requirements
- Error handling
- Security
- Maintainability
- Deployment problems

Request:
{state.get("input", "")}
"""

    return {
        "review": review
    }


# ============================================================
# NODE 4 — FINALIZER
# ============================================================

def finalizer(state: DevCrewState):

    logger.info("NODE 4: Finalizer")

    prompt = f"""
You are Dev Crew, a professional AI software development team.

USER REQUEST:
{state.get("input", "")}

PLANNER:
{state.get("plan", "")}

DEVELOPER:
{state.get("implementation", "")}

REVIEWER:
{state.get("review", "")}

Create the final response.

Use this format:

## Dev Crew Result

### Approach
Explain the solution briefly.

### Implementation
Give the required code or implementation details.

### Review
Mention important issues and improvements.

### Next Steps
Explain how to run or deploy it.

Rules:
- Be practical.
- Be concise but useful.
- Do not reveal chain-of-thought.
- Do not claim that code was tested unless it actually was.
"""

    try:

        response = llm.invoke(prompt)

        content = response.content

        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))

            content = "\n".join(parts)

        content = str(content).strip()

        if not content:
            raise RuntimeError("Gemini returned an empty response.")

        logger.info("Gemini response received")

        return {
            "output": content
        }

    except Exception as error:

        logger.exception("Gemini error")

        # IMPORTANT:
        # Never leave the Playground hanging.
        return {
            "output": f"""
## Dev Crew Result

### Status

The LangGraph workflow completed, but the Gemini generation step
returned an error.

### Error

{error}

### Workflow

Planner → Developer → Reviewer → Finalizer

The LangGraph agent itself is running correctly.

Please check the Gemini API key and Render environment variables.
"""
        }


# ============================================================
# BUILD GRAPH
# ============================================================

graph_builder = StateGraph(DevCrewState)

graph_builder.add_node("planner", planner)
graph_builder.add_node("developer", developer)
graph_builder.add_node("reviewer", reviewer)
graph_builder.add_node("finalizer", finalizer)

graph_builder.set_entry_point("planner")

graph_builder.add_edge("planner", "developer")
graph_builder.add_edge("developer", "reviewer")
graph_builder.add_edge("reviewer", "finalizer")
graph_builder.add_edge("finalizer", END)

graph = graph_builder.compile()


# ============================================================
# LANGSERVE FUNCTION
# ============================================================

def run_agent(data):

    logger.info("PLAYGROUND REQUEST RECEIVED")

    # LangServe sends a dictionary.
    if isinstance(data, dict):
        user_input = data.get("input", "")
    else:
        user_input = str(data)

    user_input = str(user_input).strip()

    if not user_input:

        return {
            "output": "Please enter a development request."
        }

    logger.info("INPUT: %s", user_input)

    try:

        result = graph.invoke(
            {
                "input": user_input
            }
        )

        return {
            "output": result.get(
                "output",
                "Dev Crew completed without producing a response."
            )
        }

    except Exception as error:

        logger.exception("GRAPH ERROR")

        return {
            "output": f"""
## Dev Crew Error

The LangGraph workflow encountered an error.

Error:
{error}
"""
        }


# ============================================================
# LANGSERVE RUNNABLE
# ============================================================

agent = RunnableLambda(run_agent)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Dev Crew LangGraph Agent",
    description="LangGraph multi-stage software development agent",
    version="1.0.0",
)


add_routes(
    app,
    agent,
    path="/agent",
)


# ============================================================
# HEALTH
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


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
