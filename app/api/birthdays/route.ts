import { auth } from "@clerk/nextjs/server";
import { db } from "@/lib/db";

// GET /api/birthdays — fetch birthday table
export async function GET() {
  try {
    const { userId: clerkId } = await auth();
    if (!clerkId) return Response.json({ success: false, error: "Unauthorized" }, { status: 401 });

    const user = await db.user.findUnique({ where: { clerkId } });
    if (!user) return Response.json({ success: true, data: [] });

    const leads = await db.policyLead.findMany({
      where: { agentId: user.id },
      select: { id: true },
    });

    // If table is empty, auto-refresh first
    const count = await db.birthdayReminder.count({
      where: { leadId: { in: leads.map(l => l.id) } },
    });

    if (count === 0) {
      await refreshBirthdays(user.id);
    }

    const reminders = await db.birthdayReminder.findMany({
      where: { leadId: { in: leads.map(l => l.id) } },
      include: {
        lead: {
          select: {
            status: true,
            issuances: { select: { policyName: true, premiumAmount: true } },
          },
        },
      },
      orderBy: { daysUntil: "asc" },
    });

    return Response.json({ success: true, data: reminders });
  } catch (error) {
    console.error("Birthday fetch error:", error);
    return Response.json({ success: false, error: "Failed to fetch birthdays" }, { status: 500 });
  }
}

// POST /api/birthdays — refresh birthday table
export async function POST() {
  try {
    const { userId: clerkId } = await auth();
    if (!clerkId) return Response.json({ success: false, error: "Unauthorized" }, { status: 401 });

    const user = await db.user.findUnique({ where: { clerkId } });
    if (!user) return Response.json({ success: false, error: "User not found" }, { status: 404 });

    const reminders = await refreshBirthdays(user.id);
    return Response.json({ success: true, data: reminders });
  } catch (error) {
    console.error("Birthday refresh error:", error);
    return Response.json({ success: false, error: "Failed to refresh" }, { status: 500 });
  }
}

// PATCH /api/birthdays — mark wish as sent
export async function PATCH(req: Request) {
  try {
    const { userId: clerkId } = await auth();
    if (!clerkId) return Response.json({ success: false, error: "Unauthorized" }, { status: 401 });

    const { id } = await req.json();
    const updated = await db.birthdayReminder.update({
      where: { id },
      data: { wishSent: true },
    });

    return Response.json({ success: true, data: updated });
  } catch (error) {
    return Response.json({ success: false, error: "Failed to update" }, { status: 500 });
  }
}

// Shared refresh logic
async function refreshBirthdays(agentId: string) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const leads = await db.policyLead.findMany({
    where: { agentId, dateOfBirth: { not: null } },
    select: { id: true, householdName: true, phone: true, dateOfBirth: true },
  });

  // Clear existing
  await db.birthdayReminder.deleteMany({
    where: { leadId: { in: leads.map(l => l.id) } },
  });

  const reminders = leads
    .filter(l => l.dateOfBirth !== null)
    .map(l => {
      const dob = new Date(l.dateOfBirth!);
      const thisYearBirthday = new Date(today.getFullYear(), dob.getMonth(), dob.getDate());

      if (thisYearBirthday < today) {
        thisYearBirthday.setFullYear(today.getFullYear() + 1);
      }

      const daysUntil = Math.round(
        (thisYearBirthday.getTime() - today.getTime()) / 86400000
      );

      return {
        leadId: l.id,
        name: l.householdName,
        phone: l.phone ?? null,
        dateOfBirth: l.dateOfBirth!,
        daysUntil,
        isToday: daysUntil === 0,
        wishSent: false,
        refreshedAt: new Date(),
      };
    })
    .sort((a, b) => a.daysUntil - b.daysUntil);

  await db.birthdayReminder.createMany({ data: reminders });
  return reminders;
}