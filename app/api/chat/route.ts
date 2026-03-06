import { BedrockRuntimeClient, ConverseCommand } from "@aws-sdk/client-bedrock-runtime";
import Groq from "groq-sdk";
import { Pinecone } from "@pinecone-database/pinecone";
import { NextRequest } from "next/server";

const bedrockClient = new BedrockRuntimeClient({
  region: process.env.AWS_REGION!,
  token: {
    token: process.env.AWS_BEARER_TOKEN_BEDROCK!,
  },
});

const groqClient = new Groq({ apiKey: process.env.GROQ_API_KEY });

const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
const index = pc.Index(process.env.PINECONE_INDEX!);

const SYSTEM_PROMPT = `You are an AI insurance assistant for rural India, helping field agents recommend the right insurance policies to households.

When an agent describes a household (income, family size, occupation, location, risks), you:
1. Briefly analyze their needs in 1-2 sentences
2. Recommend 2-3 specific insurance schemes that fit them — prioritize details from the policy documents provided in context
3. For each policy return structured data

Always respond in this exact JSON format:
{
  "analysis": "Brief analysis of household needs",
  "policies": [
    {
      "name": "Policy name",
      "provider": "Provider name",
      "premium": "Premium amount",
      "coverage": "Coverage amount",
      "tag": "Health/Crop/Life/Accident"
    }
  ]
}

If policy documents are provided in the context, use the actual premium and coverage figures from them.
Also consider real Indian government schemes like:
- Ayushman Bharat (health)
- PM Fasal Bima Yojana (crop)
- PM Jeevan Jyoti Bima Yojana (life)
- PM Suraksha Bima Yojana (accident)
- RSBY (health for BPL families)

Only respond with valid JSON, no extra text.`;

import { DenseEmbedding } from "@pinecone-database/pinecone/dist/pinecone-generated-ts-fetch/inference/models/DenseEmbedding";

async function retrieveContext(message: string): Promise<string> {
  try {
    const embeddings = await pc.inference.embed({
      model: "llama-text-embed-v2",
      inputs: [message],
      parameters: { input_type: "query", truncate: "END" },
    });

    const dense = embeddings.data[0] as unknown as DenseEmbedding;
    const vector = dense.values;

    const results = await index.query({
      vector,
      topK: 5,
      includeMetadata: true,
    });

    if (!results.matches || results.matches.length === 0) return "";

    return results.matches
      .map((m) => `[Source: ${m.metadata?.source}]\n${m.metadata?.text}`)
      .join("\n\n---\n\n");
  } catch (err) {
    console.warn("Pinecone retrieval failed:", err);
    return "";
  }
}
async function callBedrock(message: string): Promise<string> {
  const command = new ConverseCommand({
    modelId: process.env.BEDROCK_MODEL_ID!,
    system: [{ text: SYSTEM_PROMPT }],
    messages: [{ role: "user", content: [{ text: message }] }],
    inferenceConfig: { maxTokens: 1024, temperature: 0.7 },
  });
  const response = await bedrockClient.send(command);
  return response.output?.message?.content?.[0]?.text ?? "{}";
}

async function callGroq(message: string): Promise<string> {
  const response = await groqClient.chat.completions.create({
    model: "llama-3.3-70b-versatile",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: message },
    ],
    max_tokens: 1024,
    temperature: 0.7,
  });
  return response.choices[0]?.message?.content ?? "{}";
}

export async function POST(req: NextRequest) {
  try {
    const { message } = await req.json();

    // Step 1: Retrieve relevant policy chunks from Pinecone
    const context = await retrieveContext(message);

    // Step 2: Augment the user message with retrieved context
    const augmentedMessage = context
      ? `Relevant policy documents:\n\n${context}\n\n---\n\nAgent's query: ${message}`
      : message;

    // Step 3: Generate response using Bedrock (with Groq fallback)
    let text: string;
    try {
      text = await callBedrock(augmentedMessage);
      console.log("Used Bedrock");
    } catch (bedrockError) {
      console.warn("Bedrock failed, falling back to Groq:", bedrockError);
      text = await callGroq(augmentedMessage);
      console.log("Used Groq fallback");
    }

    // Strip markdown code fences if model wraps response in ```json
    const clean = text.replace(/```json|```/g, "").trim();
    const parsed = JSON.parse(clean);

    return Response.json({ success: true, data: parsed });
  } catch (error) {
    console.error("AI error:", error);
    return Response.json({ success: false, error: "AI unavailable" }, { status: 500 });
  }
}