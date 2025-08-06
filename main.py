import os
import asyncio
import json
from dotenv import load_dotenv

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.tool import FunctionTool
from typing import Any

import google.generativeai as genai  # ✅ Gemini SDK

# --- Load Environment Variables ---
load_dotenv()

# 🚫 Disable tracing for clean output (optional for beginners)
set_tracing_disabled(disabled=True)

# 🔐 1) Environment & Client Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # 🔑 Get your API key from environment
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"  # 🌐 Gemini-compatible base URL (set this in .env file)

# 🌐 Initialize the AsyncOpenAI-compatible client with Gemini details
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=BASE_URL,
)

# 🧠 2) Model Initialization
model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",        # ⚡ Fast Gemini model
    openai_client=external_client
)

class ToolOnlyModel:
    async def run(self, messages, functions=None, config=None) -> Any:
        return {
            "role": "assistant",
            "content": "This agent only uses tools. Please try asking something that requires tool use."
        }

# --- Tool: Answer Academic Questions ---
async def answer_academic_question(input: dict) -> str:
    question = input.get("question", "")
    return f"📘 Here's a helpful explanation for your question:\n{question}"

# --- Tool: Provide Study Tips ---
async def provide_study_tips(input: dict) -> str:
    topic = input.get("topic", "general").lower()
    tips = {
        "math": "1. Practice daily\n2. Understand concepts\n3. Review mistakes",
        "science": "1. Watch visual demos\n2. Use flashcards\n3. Revise regularly",
        "general": "1. Use active recall\n2. Teach others\n3. Take spaced breaks"
    }
    return tips.get(topic, tips["general"])

# --- Tool: Summarize Text ---
async def summarize_text(input: dict) -> str:
    text = input.get("text", "")
    if len(text.split()) < 5:
        return "⚠️ Text too short to summarize."
    return f"📄 Summary:\n{text[:120]}..."

# --- Tool: Gemini SDK-based API Call ---
async def call_gemini(input: dict) -> str:
    question = input.get("question", "").strip()
    if not question:
        return "⚠️ No question provided."

    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, lambda: model.generate_content(question))
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini SDK Error: {e}"

# --- Tool Wrappers ---
async def invoke_tool(fn, ctx, input_json):
    return await fn(json.loads(input_json))

# --- Register Tools ---
tools = [
    FunctionTool(
        name="answer_academic_question",
        description="Answer academic questions",
        params_json_schema={"question": {"type": "string"}},
        on_invoke_tool=lambda ctx, input_json: invoke_tool(answer_academic_question, ctx, input_json),
    ),
    FunctionTool(
        name="provide_study_tips",
        description="Give study tips for a topic",
        params_json_schema={"topic": {"type": "string"}},
        on_invoke_tool=lambda ctx, input_json: invoke_tool(provide_study_tips, ctx, input_json),
    ),
    FunctionTool(
        name="summarize_text",
        description="Summarize a passage of text",
        params_json_schema={"text": {"type": "string"}},
        on_invoke_tool=lambda ctx, input_json: invoke_tool(summarize_text, ctx, input_json),
    ),
    FunctionTool(
        name="call_gemini",
        description="Use Gemini to answer any prompt",
        params_json_schema={"question": {"type": "string"}},
        on_invoke_tool=lambda ctx, input_json: invoke_tool(call_gemini, ctx, input_json),
    ),
]

# --- Create Agent (tool-only model) ---
agent = Agent(
    name="AcademicAssistant",
    tools=tools,
    model=model
)

runner = Runner()

# --- Main Async App ---
async def main():
    print("🎓 Academic Assistant Ready!")
    print("Type your question, or 'quit' to exit.")

    while True:
        user_input = input("\n🧑 You: ")
        if user_input.lower() in {"quit", "exit"}:
            break

        try:
            result = await runner.run(agent, user_input)
            print(f"🤖 Agent: {result.final_output}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
