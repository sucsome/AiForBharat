import { BedrockRuntimeClient, ConverseCommand } from "@aws-sdk/client-bedrock-runtime";
import Groq from "groq-sdk";
import { Pinecone } from "@pinecone-database/pinecone";
import { NextRequest } from "next/server";

const bedrockClient = new BedrockRuntimeClient({
  region: process.env.AWS_REGION!,
  token: { token: process.env.AWS_BEARER_TOKEN_BEDROCK! },
});

const groqClient = new Groq({ apiKey: process.env.GROQ_API_KEY });
const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
const index = pc.Index(process.env.PINECONE_INDEX!);

const SYSTEM_PROMPT = `You are an AI insurance assistant for rural India, helping field agents recommend insurance policies to low-income households.

You have access to real insurance policy documents. Use them to give accurate, specific answers.

STRICT RULES:
- Always respond with valid JSON only. Zero plain text outside JSON.
- Never add markdown, code blocks, or extra explanation.

RULE 1 - RECOMMENDATION MODE:
Triggered when agent describes a household (income, family size, occupation, location, needs).
- ALWAYS recommend exactly 3 policies
- Use real policy names and figures from the provided documents
- Respond ONLY with:
{"type":"recommendation","analysis":"2 sentence analysis of household needs","policies":[{"name":"Exact policy name from documents","provider":"Provider name","premium":"₹X/month","coverage":"₹X lakh","tag":"Health/Crop/Life/Accident"},{"name":"...","provider":"...","premium":"...","coverage":"...","tag":"..."},{"name":"...","provider":"...","premium":"...","coverage":"...","tag":"..."}]}

RULE 2 - CONVERSATION MODE:
Triggered for follow-up questions, policy details, comparisons, greetings, anything else.
- Answer using specific details from the provided policy documents
- Give concrete numbers, facts, eligibility criteria from the documents
- Respond ONLY with:
{"type":"message","content":"Specific, detailed answer using facts from the policy documents"}`;

// ← NEW: history message type
interface HistoryMessage {
  role: "agent" | "ai";
  content: string;
}

async function retrieveContext(message: string): Promise<string> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const results = await (index as any).searchRecords({
      query: { topK: 8, inputs: { text: message } },
      fields: ["text", "source"],
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const hits = results?.result?.hits as any[];
    if (!hits || hits.length === 0) return "";

    console.log("Retrieved", hits.length, "chunks — sources:", hits.map((h: any) => h.fields?.source));

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return hits
      .map((h: any) => `[Source: ${h.fields?.source}]\n${h.fields?.text}`)
      .join("\n\n---\n\n");
  } catch (err) {
    console.error("Pinecone retrieval error:", err);
    return "";
  }
}

// ← CHANGED: accepts history, spreads it before the current message
async function callBedrock(message: string, history: HistoryMessage[]): Promise<string> {
  const command = new ConverseCommand({
    modelId: process.env.BEDROCK_MODEL_ID!,
    system: [{ text: SYSTEM_PROMPT }],
    messages: [
      ...history.slice(-20).map((m) => ({
        role: (m.role === "agent" ? "user" : "assistant") as "user" | "assistant",
        content: [{ text: m.content }],
      })),
      { role: "user", content: [{ text: message }] },
    ],
    inferenceConfig: { maxTokens: 1500, temperature: 0.2 },
  });
  const response = await bedrockClient.send(command);
  return response.output?.message?.content?.[0]?.text ?? "{}";
}

// ← CHANGED: accepts history, spreads it before the current message
async function callGroq(message: string, history: HistoryMessage[]): Promise<string> {
  const response = await groqClient.chat.completions.create({
    model: "meta-llama/llama-4-scout-17b-16e-instruct",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      ...history.slice(-20).map((m) => ({
        role: (m.role === "agent" ? "user" : "assistant") as "user" | "assistant",
        content: m.content,
      })),
      { role: "user", content: message },
    ],
    max_tokens: 1500,
    temperature: 0.2,
  });
  return response.choices[0]?.message?.content ?? "{}";
}

export async function POST(req: NextRequest) {
  try {
    // ← CHANGED: destructure history (defaults to [] so nothing breaks if not sent)
    const { message, history = [] } = await req.json() as { message: string; history: HistoryMessage[] };

    const context = await retrieveContext(message);
    const augmentedMessage = context
      ? `POLICY DOCUMENTS (use these for accurate info):\n\n${context}\n\n---\n\nAgent message: ${message}`
      : message;

    let text: string;
    try {
      text = await callBedrock(augmentedMessage, history);
      console.log("Used Bedrock");
    } catch {
      text = await callGroq(augmentedMessage, history);
      console.log("Used Groq fallback");
    }

    const clean = text.replace(/```json|```/g, "").trim();

    let parsed: { type: string; analysis?: string; policies?: unknown[]; content?: string };
    try {
      parsed = JSON.parse(clean);
    } catch {
      parsed = { type: "message", content: clean };
    }

    if (parsed.type === "recommendation" && Array.isArray(parsed.policies)) {
      if (parsed.policies.length < 2) {
        parsed = { type: "message", content: parsed.analysis ?? "Here are some policy suggestions based on the household profile." };
      }
    }

    return Response.json({ success: true, data: parsed });
  } catch (error) {
    console.error("AI error:", error);
    return Response.json({ success: false, error: "AI unavailable" }, { status: 500 });
  }
}