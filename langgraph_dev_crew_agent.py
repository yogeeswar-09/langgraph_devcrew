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
    temperature=0.2,
    max_retries=0,
    timeout=45,
)


# ============================================================
# SHARED LANGGRAPH STATE
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
# SAFE GEMINI CALL
# ============================================================

def call_gemini(prompt: str, agent_name: str) -> str:

    logger.info("%s -> Gemini", agent_name)

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

        return str(content).strip()

    except Exception as error:

        logger.exception("%s failed", agent_name)

        return (
            f"{agent_name} failed.\n\n"
            f"Error: {error}"
        )


# ============================================================
# AGENT 1
# REQUIREMENTS + ARCHITECTURE
# ============================================================

def requirements_architect_agent(
    state: DevCrewState
) -> DevCrewState:

    logger.info(
        "AGENT 1 -> Requirements + Architecture"
    )

    request = state["user_request"]

    prompt = f"""
You are the Requirements and Architecture team
inside a multi-agent software development system called Dev Crew.

USER REQUEST:

{request}

Your job has TWO responsibilities.

============================================================
PART A — REQUIREMENTS ANALYST
============================================================

Identify:

1. Functional requirements
2. Non-functional requirements
3. User roles
4. Inputs and outputs
5. Constraints
6. Edge cases

============================================================
PART B — SOFTWARE ARCHITECT
============================================================

Based on those requirements, design:

1. Technology stack
2. Application architecture
3. Main modules
4. API structure
5. Database design if required
6. Data flow
7. Authentication/security
8. Deployment approach

Return the answer using:

# Requirements

...

# Architecture

...

Do not reveal chain-of-thought.
Be practical and concise.
"""

    result = call_gemini(
        prompt,
        "Requirements + Architecture Agent"
    )

    return {
        "requirements": result,
        "architecture": result
    }


# ============================================================
# AGENT 2
# DEVELOPER + TESTING
# ============================================================

def developer_testing_agent(
    state: DevCrewState
) -> DevCrewState:

    logger.info(
        "AGENT 2 -> Developer + Testing"
    )

    prompt = f"""
You are the Developer and QA team inside Dev Crew.

USER REQUEST:

{state["user_request"]}

REQUIREMENTS AND ARCHITECTURE:

{state["requirements"]}

Your job has TWO responsibilities.

============================================================
PART A — SENIOR DEVELOPER
============================================================

Create the implementation plan.

Include:

1. Project structure
2. Important files
3. Backend/frontend implementation
4. APIs
5. Database implementation
6. Configuration
7. Error handling
8. Relevant code examples where useful

============================================================
PART B — QA ENGINEER
============================================================

Create a testing strategy.

Include:

1. Unit tests
2. API tests
3. Integration tests
4. Validation tests
5. Security tests
6. Edge cases
7. Expected behavior

Return:

# Implementation

...

# Testing Strategy

...

Do not claim anything has actually been tested.
"""

    result = call_gemini(
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

def reviewer_lead_agent(
    state: DevCrewState
) -> DevCrewState:

    logger.info(
        "AGENT 3 -> Reviewer + Lead"
    )

    prompt = f"""
You are the Senior Reviewer and Lead Engineer
of Dev Crew.

USER REQUEST:

{state["user_request"]}

REQUIREMENTS:

{state["requirements"]}

ARCHITECTURE:

{state["architecture"]}

IMPLEMENTATION:

{state["implementation"]}

TESTING:

{state["testing"]}

You have TWO responsibilities.

============================================================
PART A — SENIOR REVIEWER
============================================================

Review the proposed solution for:

1. Correctness
2. Requirement coverage
3. Security
4. Error handling
5. Maintainability
6. Scalability
7. Testing coverage
8. Deployment readiness

Identify specific improvements.

============================================================
PART B — LEAD ENGINEER
============================================================

Create the final professional Dev Crew report.

Use EXACTLY this structure:

# Dev Crew Final Report

## 1. Project Understanding

Explain what needs to be built.

## 2. Requirements

Summarize the key requirements.

## 3. Architecture

Explain the architecture and technology stack.

## 4. Implementation

Explain the implementation approach and
provide useful code/project structure where appropriate.

## 5. Testing Strategy

Explain how the application should be tested.

## 6. Technical Review

Summarize risks, weaknesses and improvements.

## 7. Deployment

Explain how the solution can be deployed.

## 8. Final Recommendation

Give a professional conclusion.

IMPORTANT:

- Do not reveal chain-of-thought.
- Do not claim code was executed.
- Do not invent test results.
- Clearly distinguish recommendations from completed work.
- Make the report understandable to a student developer.

At the end, include:

### Reviewer Verdict

State whether the proposed solution is:

READY

or

NEEDS IMPROVEMENT

and briefly explain why.
"""

    result = call_gemini(
        prompt,
        "Reviewer + Lead Agent"
    )

    return {
        "review": result,
        "final_report": result
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(DevCrewState)

builder.add_node(
    "requirements_architect",
    requirements_architect_agent
)

builder.add_node(
    "developer_testing",
    developer_testing_agent
)

builder.add_node(
    "reviewer_lead",
    reviewer_lead_agent
)


# ============================================================
# WORKFLOW
# ============================================================

builder.set_entry_point(
    "requirements_architect"
)

builder.add_edge(
    "requirements_architect",
    "developer_testing"
)

builder.add_edge(
    "developer_testing",
    "reviewer_lead"
)

builder.add_edge(
    "reviewer_lead",
    END
)


dev_crew_graph = builder.compile()


# ============================================================
# LANGSERVE
# ============================================================

def run_dev_crew(
    user_request: str
) -> str:

    if not isinstance(user_request, str):
        user_request = str(user_request)

    user_request = user_request.strip()

    if not user_request:
        return "Please enter a software development request."

    logger.info(
        "DEV CREW REQUEST: %s",
        user_request
    )

    try:

        result = dev_crew_graph.invoke(
            {
                "user_request": user_request
            }
        )

        return result.get(
            "final_report",
            "Dev Crew completed without a final report."
        )

    except Exception as error:

        logger.exception(
            "Dev Crew workflow failed"
        )

        return (
            "# Dev Crew Error\n\n"
            f"{error}"
        )


agent = RunnableLambda(
    run_dev_crew
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Dev Crew - LangGraph Multi-Agent",
    version="3.0.0",
    description=(
        "Quota-efficient multi-agent software "
        "development team implemented with LangGraph."
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
    output_type=str,
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

        "agents": [
            "Requirements + Architecture Agent",
            "Developer + Testing Agent",
            "Reviewer + Lead Agent"
        ],

        "workflow": (
            "Requirements/Architecture -> "
            "Developer/Testing -> "
            "Reviewer/Lead"
        ),

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
        "framework": "LangGraph",
        "gemini_calls_per_request": 3
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
