// app/api/ocr/route.ts
import { NextRequest } from "next/server";
import { auth } from "@clerk/nextjs/server";
import Groq from "groq-sdk";

const groqClient = new Groq({ apiKey: process.env.GROQ_API_KEY });

export async function POST(req: NextRequest) {
  const { userId } = await auth();
  if (!userId) return Response.json({ success: false, error: "Unauthorized" }, { status: 401 });

  const { imageBase64, mediaType, expectedDocType, householdName } = await req.json() as {
    imageBase64: string;
    mediaType: string;
    expectedDocType: "Aadhaar Card" | "PAN Card";
    householdName: string;
  };

  if (!imageBase64 || !expectedDocType || !householdName) {
    return Response.json({ success: false, error: "Missing fields" }, { status: 400 });
  }

  const prompt = `You are a KYC document validator for Indian insurance.

Expected document: ${expectedDocType}
Customer name on policy: "${householdName}"

Look at the image and respond ONLY with this JSON:
{
  "isCorrectDoc": true or false,
  "detectedDocType": "what you see (e.g. Aadhaar Card, PAN Card, Driving License)",
  "extractedName": "name printed on the document exactly as shown",
  "nameMatches": true or false,
  "isReadable": true or false,
  "confidence": "high" or "medium" or "low"
}

For nameMatches: use fuzzy matching — allow Hindi/regional transliteration differences, minor spelling variations, first name only is acceptable. If reasonably similar, mark true.
If the image is blurry or not a valid ID, set isReadable to false.`;

  try {
    const completion = await groqClient.chat.completions.create({
      model: "meta-llama/llama-4-scout-17b-16e-instruct",
      max_tokens: 200,
      temperature: 0,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image_url",
              image_url: { url: `data:${mediaType};base64,${imageBase64}` },
            },
            { type: "text", text: prompt },
          ],
        },
      ],
    });

    const raw = completion.choices[0]?.message?.content ?? "{}";
    const clean = raw.replace(/```json|```/g, "").trim();

    let result: {
      isCorrectDoc: boolean;
      detectedDocType: string;
      extractedName: string;
      nameMatches: boolean;
      isReadable: boolean;
      confidence: string;
    };

    try {
      result = JSON.parse(clean);
    } catch {
      return Response.json({ success: true, valid: false, reason: "Could not read document. Please upload a clearer image." });
    }

    const valid =
      result.isReadable &&
      result.isCorrectDoc &&
      result.nameMatches &&
      result.confidence !== "low";

    const reason = !result.isReadable
      ? "Document is unreadable. Please upload a clearer photo."
      : !result.isCorrectDoc
      ? `Wrong document — got ${result.detectedDocType}, expected ${expectedDocType}.`
      : !result.nameMatches
      ? `Name "${result.extractedName}" doesn't match customer "${householdName}".`
      : result.confidence === "low"
      ? "Image quality too low. Please retake the photo."
      : `✓ ${result.detectedDocType} verified for ${result.extractedName}`;

    return Response.json({
      success: true,
      valid,
      detectedDocType: result.detectedDocType,
      extractedName: result.extractedName,
      reason,
    });
  } catch (err) {
    console.error("OCR error:", err);
    return Response.json({ success: false, error: "Validation failed" }, { status: 500 });
  }
}