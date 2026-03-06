"use client";

import { useUser } from "@clerk/nextjs";
import { redirect } from "next/navigation";
import { useState } from "react";
import Sidebar from "@/components/dashboard/Sidebar";
import ChatArea from "@/components/dashboard/ChatArea";

interface Lead {
  id: string;
  householdName: string;
  lastMessage: string;
  time: string;
  unread: number;
}

export default function DashboardPage() {
  const { user, isLoaded } = useUser();
  const [activeLead, setActiveLead] = useState<Lead | null>(null);

  if (isLoaded && !user) redirect("/sign-in");

  return (
    <div className="flex h-screen bg-slate-50">
      <Sidebar
        user={{ name: user?.firstName ?? "Agent", avatar: user?.imageUrl ?? "" }}
        activeLead={activeLead}
        onSelectLead={setActiveLead}
      />
      {activeLead ? (
        <ChatArea lead={activeLead} />
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center text-center gap-3">
          <div className="w-12 h-12 bg-emerald-100 rounded-full flex items-center justify-center">
            <span className="text-2xl">🏡</span>
          </div>
          <p className="text-sm font-semibold text-slate-700">No household selected</p>
          <p className="text-xs text-slate-400">Click + to start a new conversation</p>
        </div>
      )}
    </div>
  );
}