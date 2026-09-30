import json
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.rag import search_documents


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL = "openai/gpt-oss-20b"
UGC_PDF = "UGC-Autonomous-College-Regulations-2023.pdf"
ACT_PDF = "THE-TAMILNADU-PRIVATE-COLLEGES-ACT.pdf"


def _document_agent(question, source):
	results = search_documents(question, source=source)
	if not results:
		return f"No matching information found in {source}."

	context = "\n\n".join(
		f"[{item['source']}, page {item['page']}]\n{item['text']}"
		for item in results
	)
	response = Groq().chat.completions.create(
		model=MODEL,
		messages=[
			{
				"role": "system",
				"content": "Answer only from the supplied document text. Say if it is not enough.",
			},
			{"role": "user", "content": f"{question}\n\n{context}"},
		],
	)
	answer = response.choices[0].message.content
	pages = sorted({item["page"] for item in results})
	citations = ", ".join(f"page {page}" for page in pages)
	return f"{answer}\nSource: {source}, {citations}"


def ugc_agent(question):
	return _document_agent(question, UGC_PDF)


def tamil_nadu_act_agent(question):
	return _document_agent(question, ACT_PDF)


def general_agent(question):
	response = Groq().chat.completions.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": "Answer this general question clearly and briefly."},
			{"role": "user", "content": question},
		],
	)
	return response.choices[0].message.content


TOOLS = [
	{
		"type": "function",
		"function": {
			"name": "ugc_agent",
			"description": "Use for questions about the UGC Autonomous College Regulations 2023 PDF.",
			"parameters": {
				"type": "object",
				"properties": {"question": {"type": "string"}},
				"required": ["question"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "tamil_nadu_act_agent",
			"description": "Use for questions about the Tamil Nadu Private Colleges Act PDF.",
			"parameters": {
				"type": "object",
				"properties": {"question": {"type": "string"}},
				"required": ["question"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "general_agent",
			"description": "Use for questions unrelated to either supplied PDF.",
			"parameters": {
				"type": "object",
				"properties": {"question": {"type": "string"}},
				"required": ["question"],
			},
		},
	},
]

AGENTS = {
	"ugc_agent": ugc_agent,
	"tamil_nadu_act_agent": tamil_nadu_act_agent,
	"general_agent": general_agent,
}


def answer_question(question):
	client = Groq()
	sources = set()
	messages = [
		{
			"role": "system",
			"content": (
				"You are an orchestrator. Choose the right agent based on the question's meaning, "
				"not keyword rules. Call the UGC or Tamil Nadu Act agent for questions about "
				"those documents, the general agent for unrelated questions, and both document "
				"agents when a question needs both. Use agent results to give a concise answer."
			),
		},
		{"role": "user", "content": question},
	]

	for _ in range(4):
		response = client.chat.completions.create(
			model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto"
		)
		message = response.choices[0].message
		if not message.tool_calls:
			answer = message.content or ""
			if sources:
				answer += "\n\nSources:\n" + "\n".join(
					f"- {source}" for source in sorted(sources)
				)
			return answer

		messages.append(message)
		for tool_call in message.tool_calls:
			arguments = json.loads(tool_call.function.arguments)
			agent_name = tool_call.function.name
			result = AGENTS[agent_name](**arguments)
			if agent_name != "general_agent" and "\nSource: " in result:
				sources.add(result.rsplit("\nSource: ", 1)[1])
			messages.append(
				{
					"role": "tool",
					"tool_call_id": tool_call.id,
					"content": result,
				}
			)

	return "I could not finish answering within the agent step limit."
