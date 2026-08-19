# 🧑‍💻 Dev Crew — Multi-Agent AI Software Engineering Workflow

> A LangGraph-powered multi-agent AI system that simulates a software engineering team to analyze project requirements, design architecture, review technical decisions, and generate a structured development report.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-1C7ED6?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C7C54?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google%20Gemini-LLM-4285F4?style=for-the-badge&logo=google)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=black)

---

## 🌟 Overview

**Dev Crew** is a multi-agent AI software engineering assistant built using **LangGraph** and **LangChain**.

The system takes a software project requirement and passes it through multiple specialized AI agents. Each agent focuses on a different aspect of the development process before the results are combined into a final technical report.

Instead of relying on a single AI response, Dev Crew follows a structured **multi-agent workflow** similar to how a real software engineering team collaborates.

---

## 🎯 What Dev Crew Does

Given a project requirement, Dev Crew can analyze:

- 📋 Project requirements
- 🏗️ System architecture
- 🧑‍💻 Implementation strategy
- 🗄️ Database design
- 🔐 Security considerations
- 🧪 Testing strategy
- 📈 Scalability
- ☁️ Deployment strategy
- 🔍 Technical risks
- 💡 Improvement recommendations

The final output is presented as a consolidated technical report.

---

# 🧠 Multi-Agent Architecture

```text
                         USER REQUIREMENT
                                │
                                ▼
                       ┌─────────────────┐
                       │   LEAD AGENT    │
                       │   COORDINATOR   │
                       └────────┬────────┘
                                │
               ┌────────────────┼────────────────┐
               │                │                │
               ▼                ▼                ▼
        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
        │  ARCHITECT  │  │  DEVELOPER  │  │  REVIEWER   │
        │    AGENT    │  │    AGENT    │  │    AGENT    │
        └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
               │                │                │
               └────────────────┼────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ LEAD SYNTHESIS  │
                       └────────┬────────┘
                                │
                                ▼
                      FINAL TECHNICAL REPORT
