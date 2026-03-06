import { auth } from "@clerk/nextjs/server";
import { NextRequest } from "next/server";
import { db } from "@/lib/db";

export async function DELETE(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const { userId: clerkId } = await auth();
    if (!clerkId) return Response.json({ success: false, error: "Unauthorized" }, { status: 401 });

    const user = await db.user.findUnique({ where: { clerkId } });
    if (!user) return Response.json({ success: false, error: "User not found" }, { status: 404 });

    // Make sure the lead belongs to this agent
    const lead = await db.policyLead.findFirst({
      where: { id: params.id, agentId: user.id },
    });
    if (!lead) return Response.json({ success: false, error: "Lead not found" }, { status: 404 });

    await db.policyLead.delete({ where: { id: params.id } });

    return Response.json({ success: true });
  } catch (error) {
    console.error("Delete lead error:", error);
    return Response.json({ success: false, error: "Failed to delete lead" }, { status: 500 });
  }
}