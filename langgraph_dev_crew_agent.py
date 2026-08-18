import os
import logging
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("devcrew")


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY or GEMINI_API_KEY is missing from Render Environment Variables."
    )


# ============================================================
# INPUT / OUTPUT SCHEMAS
#
# THIS IS THE IMPORTANT FIX FOR LANGSERVE PLAYGROUND
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        ...,
        description="Software development task for the Dev Crew"
    )


class AgentOutput(BaseModel):
    output: str


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
# GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    max_retries=1,
    timeout=30,
)


# ============================================================
# NODE 1
# PLANNER
# ============================================================

def planner(state: DevCrewState):

    logger.info("DEV CREW -> PLANNER")

    task = state.get("input", "").strip()

    plan = f"""
Task received:

{task}

Development plan:

1. Understand the requested application.
2. Identify functional requirements.
3. Identify the required technologies.
4. Design the application architecture.
5. Plan the implementation.
6. Consider testing, security and deployment.
"""

    return {
        "plan": plan
    }


# ============================================================
# NODE 2
# DEVELOPER
# ============================================================

def developer(state: DevCrewState):

    logger.info("DEV CREW -> DEVELOPER")

    implementation = f"""
Developer analysis:

Task:
{state.get("input", "")}

Plan:
{state.get("plan", "")}

Implementation should contain:

- Project structure
- Backend/frontend components when applicable
- APIs
- Database considerations
- Error handling
- Configuration
- Deployment considerations
"""

    return {
        "implementation": implementation
    }


# ============================================================
# NODE 3
# REVIEWER
# ============================================================

def reviewer(state: DevCrewState):

    logger.info("DEV CREW -> REVIEWER")

    review = f"""
Technical review:

Task:
{state.get("input", "")}

Developer proposal:
{state.get("implementation", "")}

Review the proposed solution for:

- Correctness
- Missing requirements
- Security
- Error handling
- Maintainability
- Scalability
- Deployment issues
"""

    return {
        "review": review
    }


# ============================================================
# NODE 4
# FINALIZER
# ============================================================

def finalizer(state: DevCrewState):

    logger.info("DEV CREW -> FINALIZER")

    prompt = f"""
You are Dev Crew, an AI software development team.

You received the following software development task:

USER TASK:
{state.get("input", "")}

PLANNER:
{state.get("plan", "")}

DEVELOPER:
{state.get("implementation", "")}

REVIEWER:
{state.get("review", "")}

Create a professional final answer.

Use this structure:

# Dev Crew Result

## 1. Understanding
Explain what the user wants.

## 2. Architecture
Explain the proposed architecture.

## 3. Implementation
Provide useful implementation details or code where appropriate.

## 4. Technical Review
Mention important considerations, risks and improvements.

## 5. How to Run
Explain how the project can be run.

## 6. Next Steps
Give practical next steps.

Do not reveal hidden chain-of-thought.
Do not claim something was tested unless it actually was.
"""


    try:

        response = llm.invoke(prompt)

        content = response.content

        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):

                    text_parts.append(
                        str(item.get("text", ""))
                    )

                else:

                    text_parts.append(str(item))

            content = "\n".join(text_parts)

        content = str(content).strip()

        if not content:

            content = "Dev Crew completed but Gemini returned an empty response."

        return {
            "output": content
        }


    except Exception as error:

        logger.exception("Gemini error")

        return {
            "output": (
                "# Dev Crew\n\n"
                "The LangGraph workflow completed, "
                "but the Gemini generation step failed.\n\n"
                f"**Error:** `{error}`"
            )
        }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(DevCrewState)


builder.add_node(
    "planner",
    planner
)

builder.add_node(
    "developer",
    developer
)

builder.add_node(
    "reviewer",
    reviewer
)

builder.add_node(
    "finalizer",
    finalizer
)


# ============================================================
# GRAPH FLOW
# ============================================================

builder.set_entry_point("planner")

builder.add_edge(
    "planner",
    "developer"
)

builder.add_edge(
    "developer",
    "reviewer"
)

builder.add_edge(
    "reviewer",
    "finalizer"
)

builder.add_edge(
    "finalizer",
    END
)


graph = builder.compile()


# ============================================================
# LANGSERVE RUNNER
# ============================================================

def run_devcrew(data: AgentInput):

    logger.info("LANGSERVE REQUEST RECEIVED")

    user_input = data.input.strip()

    if not user_input:

        return AgentOutput(
            output="Please enter a software development task."
        )


    logger.info(
        "USER INPUT: %s",
        user_input
    )


    try:

        result = graph.invoke(
            {
                "input": user_input
            }
        )


        return AgentOutput(
            output=result.get(
                "output",
                "Dev Crew completed without producing an output."
            )
        )


    except Exception as error:

        logger.exception(
            "LANGGRAPH ERROR"
        )

        return AgentOutput(
            output=(
                "# Dev Crew Error\n\n"
                f"`{error}`"
            )
        )


# ============================================================
# CREATE TYPED LANGSERVE RUNNABLE
#
# THIS IS THE CRITICAL PART
# ============================================================

agent = RunnableLambda(
    run_devcrew
).with_types(
    input_type=AgentInput,
    output_type=AgentOutput
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Dev Crew - LangGraph Agent",
    description=(
        "A LangGraph-based AI software development crew "
        "consisting of Planner, Developer, Reviewer and Finalizer."
    ),
    version="1.0.0",
)


# ============================================================
# LANGSERVE ROUTES
# ============================================================

add_routes(
    app,
    agent,
    path="/agent"
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
# START SERVER
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
