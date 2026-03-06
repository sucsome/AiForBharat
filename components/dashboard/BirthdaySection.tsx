// Drop-in replacement for the birthdays section inside crm/page.tsx
// Replace the entire {activePage === "birthdays" && (...)} block with this component

import { useState, useEffect } from "react";
import { Phone, RefreshCw, CheckCircle2, Loader2 } from "lucide-react";

interface BirthdayEntry {
  id: string;
  name: string;
  phone: string | null;
  dateOfBirth: string;
  daysUntil: number;
  isToday: boolean;
  wishSent: boolean;
  refreshedAt: string;
  lead: {
    status: string;
    issuances: { policyName: string; premiumAmount: number | null }[];
  };
}

function getAge(dob: string) {
  return new Date().getFullYear() - new Date(dob).getFullYear();
}

function formatDOB(dob: string) {
  return new Date(dob).toLocaleDateString("en-IN", { day: "numeric", month: "short" });
}

export function BirthdaySection() {
  const [birthdays, setBirthdays] = useState<BirthdayEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [tab, setTab] = useState<"today" | "week" | "upcoming">("today");

  useEffect(() => { fetchBirthdays(); }, []);

  const fetchBirthdays = async () => {
    try {
      const res = await fetch("/api/birthdays");
      const json = await res.json();
      if (json.success) setBirthdays(json.data);
    } catch { console.error("Failed to fetch birthdays"); }
    finally { setLoading(false); }
  };

  const refresh = async () => {
    setRefreshing(true);
    try {
      await fetch("/api/birthdays", { method: "POST" });
      await fetchBirthdays();
    } finally { setRefreshing(false); }
  };

  const markWishSent = async (id: string) => {
    await fetch("/api/birthdays", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    setBirthdays(prev => prev.map(b => b.id === id ? { ...b, wishSent: true } : b));
  };

  const todayList = birthdays.filter(b => b.daysUntil === 0);
  const weekList = birthdays.filter(b => b.daysUntil > 0 && b.daysUntil <= 7);
  const upcomingList = birthdays.filter(b => b.daysUntil > 7);

  const activeList = tab === "today" ? todayList : tab === "week" ? weekList : upcomingList;

  const lastRefreshed = birthdays[0]?.refreshedAt
    ? new Date(birthdays[0].refreshedAt).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })
    : null;

  return (
    <div className="space-y-4">
      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Birthdays Today", value: todayList.length, sub: "Action needed", color: "text-pink-600" },
          { label: "This Week", value: weekList.length, sub: "Upcoming", color: "text-orange-500" },
          { label: "Wishes Sent", value: birthdays.filter(b => b.wishSent).length, sub: "Today", color: "text-emerald-600" },
        ].map(s => (
          <div key={s.label} className="bg-white rounded-2xl p-4 border border-slate-100 shadow-sm">
            <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
            <p className="text-sm font-medium text-slate-700 mt-1">{s.label}</p>
            <p className="text-xs text-slate-400">{s.sub}</p>
          </div>
        ))}
      </div>

      {/* Tabs + Refresh */}
      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          {[
            { key: "today", label: `Today (${todayList.length})` },
            { key: "week", label: `This Week (${weekList.length})` },
            { key: "upcoming", label: `Upcoming (${upcomingList.length})` },
          ].map(t => (
            <button key={t.key} onClick={() => setTab(t.key as typeof tab)}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
                tab === t.key ? "bg-slate-900 text-white" : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-50"
              }`}>
              {t.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          {lastRefreshed && <p className="text-xs text-slate-400">Refreshed {lastRefreshed}</p>}
          <button onClick={refresh} disabled={refreshing}
            className="flex items-center gap-1.5 text-xs bg-white border border-slate-200 hover:bg-slate-50 text-slate-600 px-3 py-2 rounded-xl transition-colors">
            {refreshing ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
            Refresh
          </button>
        </div>
      </div>

      {/* Table */}
      {loading ? (
        <div className="bg-white rounded-2xl border border-slate-100 p-8 flex justify-center">
          <Loader2 className="w-5 h-5 text-emerald-500 animate-spin" />
        </div>
      ) : activeList.length === 0 ? (
        <div className="bg-white rounded-2xl p-10 text-center border border-slate-100">
          <p className="text-3xl mb-2">🎂</p>
          <p className="text-slate-500">No birthdays {tab === "today" ? "today" : tab === "week" ? "this week" : "upcoming"}</p>
        </div>
      ) : (
        <div className="bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden">
          {/* Table Header */}
          <div className="grid grid-cols-6 gap-4 px-4 py-3 bg-slate-50 border-b border-slate-100 text-xs font-semibold text-slate-500 uppercase tracking-wide">
            <div className="col-span-2">Customer</div>
            <div>Date of Birth</div>
            <div>Policy</div>
            <div>Days Until</div>
            <div>Action</div>
          </div>

          {/* Table Rows */}
          <div className="divide-y divide-slate-50">
            {activeList.map(b => (
              <div key={b.id} className={`grid grid-cols-6 gap-4 px-4 py-3 items-center hover:bg-slate-50 transition-colors ${b.wishSent ? "opacity-60" : ""}`}>
                {/* Name */}
                <div className="col-span-2 flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-pink-100 flex items-center justify-center shrink-0">
                    <span className="text-pink-700 font-bold text-xs">{b.name[0]}</span>
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-slate-900 flex items-center gap-1">
                      {b.name}
                      {b.isToday && <span>🎂</span>}
                      {b.wishSent && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />}
                    </p>
                    {b.phone && <p className="text-xs text-slate-400">{b.phone}</p>}
                  </div>
                </div>

                {/* DOB */}
                <div>
                  <p className="text-sm text-slate-700">{formatDOB(b.dateOfBirth)}</p>
                  <p className="text-xs text-slate-400">Turning {getAge(b.dateOfBirth)}</p>
                </div>

                {/* Policy */}
                <div>
                  {b.lead.issuances.length > 0 ? (
                    <span className="text-xs bg-emerald-50 text-emerald-700 px-2 py-1 rounded-lg">
                      {b.lead.issuances[0].policyName.split(" ").slice(0, 3).join(" ")}
                    </span>
                  ) : (
                    <span className="text-xs text-slate-400">No policy</span>
                  )}
                </div>

                {/* Days Until */}
                <div>
                  {b.daysUntil === 0 ? (
                    <span className="text-xs font-bold bg-pink-100 text-pink-600 px-2 py-1 rounded-full">Today! 🎉</span>
                  ) : (
                    <span className={`text-xs font-semibold px-2 py-1 rounded-full ${
                      b.daysUntil <= 7 ? "bg-orange-50 text-orange-600" : "bg-blue-50 text-blue-600"
                    }`}>
                      {b.daysUntil} days
                    </span>
                  )}
                </div>

                {/* Actions */}
                <div className="flex gap-1.5">
                  <button
                    onClick={() => markWishSent(b.id)}
                    disabled={b.wishSent}
                    className={`text-xs px-2.5 py-1.5 rounded-lg font-medium transition-colors ${
                      b.wishSent
                        ? "bg-slate-100 text-slate-400 cursor-default"
                        : "bg-blue-600 hover:bg-blue-700 text-white"
                    }`}
                  >
                    {b.wishSent ? "Sent ✓" : "💬 Wish"}
                  </button>
                  {b.phone && (
                    <a href={`tel:${b.phone}`}
                      className="text-xs px-2.5 py-1.5 rounded-lg font-medium border border-slate-200 hover:bg-slate-50 text-slate-600 flex items-center gap-1 transition-colors">
                      <Phone className="w-3 h-3" />
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}