import { BedrockRuntimeClient, ConverseCommand } from "@aws-sdk/client-bedrock-runtime";
import Groq from "groq-sdk";
import { NextRequest } from "next/server";

const bedrockClient = new BedrockRuntimeClient({
  region: process.env.AWS_REGION!,
  token: {
    token: process.env.AWS_BEARER_TOKEN_BEDROCK!,
  },
});

const groqClient = new Groq({ apiKey: process.env.GROQ_API_KEY });

const SYSTEM_PROMPT = `You are an AI insurance assistant for rural India, helping field agents recommend the right insurance policies to households.

When an agent describes a household (income, family size, occupation, location, risks), you:
1. Briefly analyze their needs in 1-2 sentences
2. Recommend 2-3 specific government or private insurance schemes that fit them
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

Focus on real Indian government schemes like:
- Ayushman Bharat (health)
- PM Fasal Bima Yojana (crop)
- PM Jeevan Jyoti Bima Yojana (life)
- PM Suraksha Bima Yojana (accident)
- RSBY (health for BPL families)

Only respond with valid JSON, no extra text.`;

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

    let text: string;

    // Try Bedrock first, fall back to Groq if it fails
    try {
      text = await callBedrock(message);
      console.log("Used Bedrock");
    } catch (bedrockError) {
      console.warn("Bedrock failed, falling back to Groq:", bedrockError);
      text = await callGroq(message);
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