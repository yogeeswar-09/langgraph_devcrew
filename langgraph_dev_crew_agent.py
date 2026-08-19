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


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dev_crew")

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

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
    temperature=0,
    max_retries=0,
    timeout=30,
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class DevCrewState(TypedDict, total=False):

    user_request: str

    requirements: str
    architecture: str

    implementation: str
    testing: str

    review: str

    final_report: str


# ============================================================
# GEMINI HELPER
# ============================================================

def ask_gemini(
    prompt: str,
    agent_name: str
) -> str:

    logger.info(
        "Running %s",
        agent_name
    )

    try:

        response = llm.invoke(prompt)

        content = response.content

        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):

                    text = item.get("text")

                    if text:
                        text_parts.append(
                            str(text)
                        )

                else:

                    text_parts.append(
                        str(item)
                    )

            content = "\n".join(text_parts)

        return str(content).strip()

    except Exception as error:

        logger.exception(
            "%s failed",
            agent_name
        )

        return (
            f"{agent_name} failed.\n"
            f"Error: {error}"
        )


# ============================================================
# AGENT 1
# REQUIREMENTS + ARCHITECTURE
# ============================================================

def requirements_architecture(
    state: DevCrewState
) -> DevCrewState:

    request = state["user_request"]

    prompt = f"""
You are Agent 1 of Dev Crew.

ROLE:
Requirements Analyst + Software Architect

USER REQUEST:
{request}

Analyze the request and produce a concise technical specification.

Return ONLY these sections:

REQUIREMENTS:
- Functional requirements
- Non-functional requirements
- Main users
- Important inputs/outputs

ARCHITECTURE:
- Recommended stack
- Main components
- API approach
- Database approach
- Authentication/security
- Deployment approach

Keep the response under 700 words.

Do not reveal chain-of-thought.
"""

    result = ask_gemini(
        prompt,
        "Requirements + Architecture Agent"
    )

    return {
        "requirements": result,
        "architecture": result
    }


# ============================================================
# AGENT 2
# DEVELOPMENT + TESTING
# ============================================================

def development_testing(
    state: DevCrewState
) -> DevCrewState:

    specification = state["requirements"]

    prompt = f"""
You are Agent 2 of Dev Crew.

ROLE:
Senior Developer + QA Engineer

PROJECT SPECIFICATION:
{specification}

Create a practical implementation and testing plan.

Return ONLY:

IMPLEMENTATION:
- Project structure
- Important modules
- API implementation
- Database implementation
- Authentication
- Error handling
- Deployment configuration

TESTING:
- Unit tests
- API tests
- Integration tests
- Security tests
- Edge cases

Keep the response under 800 words.

Do not claim that anything was actually executed or tested.
Do not reveal chain-of-thought.
"""

    result = ask_gemini(
        prompt,
        "Developer + Testing Agent"
    )

    return {
        "implementation": result,
        "testing": result
    }


# ============================================================
# AGENT 3
# REVIEWER + LEAD
# ============================================================

def reviewer_lead(
    state: DevCrewState
) -> DevCrewState:

    # --------------------------------------------------------
    # Create a compact input instead of sending huge outputs.
    # --------------------------------------------------------

    requirements = state.get(
        "requirements",
        ""
    )[:4500]

    implementation = state.get(
        "implementation",
        ""
    )[:5000]

    prompt = f"""
You are Agent 3 of Dev Crew.

ROLE:
Senior Reviewer + Lead Engineer

USER REQUEST:
{state["user_request"]}

PROJECT SPECIFICATION:
{requirements}

DEVELOPMENT PLAN:
{implementation}

Review the proposed solution and create the final report.

Evaluate:

1. Requirement coverage
2. Architecture quality
3. Implementation quality
4. Security
5. Testing
6. Maintainability
7. Scalability
8. Deployment readiness

Return EXACTLY this structure:

# Dev Crew Final Report

## Project Understanding

## Requirements

## Architecture

## Implementation

## Testing Strategy

## Technical Review

## Deployment

## Final Recommendation

## Reviewer Verdict

Choose:
READY
or
NEEDS IMPROVEMENT

Keep the final answer under 1200 words.

Do not claim code was executed.
Do not invent test results.
Do not reveal chain-of-thought.
"""

    result = ask_gemini(
        prompt,
        "Reviewer + Lead Agent"
    )

    return {
        "review": result,
        "final_report": result
    }


# ============================================================
# LANGGRAPH
# ============================================================

builder = StateGraph(
    DevCrewState
)

builder.add_node(
    "requirements_architecture",
    requirements_architecture
)

builder.add_node(
    "development_testing",
    development_testing
)

builder.add_node(
    "reviewer_lead",
    reviewer_lead
)


# ============================================================
# GRAPH FLOW
# ============================================================

builder.set_entry_point(
    "requirements_architecture"
)

builder.add_edge(
    "requirements_architecture",
    "development_testing"
)

builder.add_edge(
    "development_testing",
    "reviewer_lead"
)

builder.add_edge(
    "reviewer_lead",
    END
)


dev_crew_graph = builder.compile()


# ============================================================
# LANGSERVE FUNCTION
# ============================================================

def run_dev_crew(
    user_request: str
) -> str:

    if user_request is None:

        return "Please enter a project requirement."

    user_request = str(
        user_request
    ).strip()

    if not user_request:

        return "Please enter a project requirement."

    logger.info(
        "Starting Dev Crew workflow"
    )

    try:

        result = dev_crew_graph.invoke(
            {
                "user_request": user_request
            }
        )

        final_report = result.get(
            "final_report"
        )

        if final_report:

            return final_report

        return (
            "# Dev Crew\n\n"
            "The workflow completed, "
            "but no final report was generated."
        )

    except Exception as error:

        logger.exception(
            "Dev Crew workflow failed"
        )

        return (
            "# Dev Crew Error\n\n"
            f"{error}"
        )


# ============================================================
# LANGSERVE RUNNABLE
# ============================================================

agent = RunnableLambda(
    run_dev_crew
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Dev Crew",
    version="4.0.0",
    description=(
        "Quota-efficient multi-agent "
        "software development workflow "
        "built with LangGraph."
    )
)


# ============================================================
# LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    agent,
    path="/agent",
    input_type=str,
    output_type=str
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "agent": "Dev Crew",
        "framework": "LangGraph",
        "type": "Multi-Agent",
        "status": "running",

        "workflow": [
            "Requirements + Architecture",
            "Development + Testing",
            "Reviewer + Lead"
        ],

        "gemini_calls_per_request": 3,

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

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                "8000"
            )
        )
    )
