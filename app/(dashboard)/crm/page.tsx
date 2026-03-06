"use client";

import { useState, useEffect } from "react";
import { useUser } from "@clerk/nextjs";
import { BirthdaySection } from "@/components/dashboard/BirthdaySection";
import Link from "next/link";
import {
  LayoutDashboard, Gift, AlertTriangle, FileText,
  ShieldCheck, Users, Bell, TrendingUp, Phone,
  Search, Filter, ChevronDown, X, Loader2,
  CheckCircle2, Calendar, Plus, ArrowUpDown,
  MessageSquare, Home
} from "lucide-react";

type LeadStatus = "NEW" | "CONTACTED" | "POLICY_ISSUED" | "REJECTED";
type ReminderType = "FOLLOWUP" | "BIRTHDAY";
type ActivePage = "dashboard" | "birthdays" | "premiums" | "leads";

interface Reminder {
  id: string;
  type: ReminderType;
  scheduledAt: string;
  note: string | null;
  isDone: boolean;
}

interface Issuance {
  id: string;
  policyName: string;
  policyProvider: string | null;
  premiumAmount: number | null;
  status: string;
  expiresAt: string | null;
}

interface Lead {
  id: string;
  householdName: string;
  phone: string | null;
  income: number | null;
  familySize: number | null;
  status: LeadStatus;
  notes: string | null;
  dateOfBirth: string | null;
  followUpAt: string | null;
  createdAt: string;
  issuances: Issuance[];
  reminders: Reminder[];
  isBirthdayToday: boolean;
  isBirthdayThisWeek: boolean;
  hasPremiumDue: boolean;
  hasPremiumDueUrgent: boolean;
  premiumsDue: Issuance[];
}

interface Stats {
  total: number;
  birthdaysToday: number;
  birthdaysThisWeek: number;
  premiumsDueCount: number;
  premiumsDueUrgentCount: number;
  policiesIssued: number;
  activeReminders: number;
}

const STATUS_COLORS: Record<LeadStatus, string> = {
  NEW: "bg-blue-100 text-blue-700",
  CONTACTED: "bg-yellow-100 text-yellow-700",
  POLICY_ISSUED: "bg-emerald-100 text-emerald-700",
  REJECTED: "bg-red-100 text-red-600",
};

function daysUntil(dateStr: string) {
  const diff = Math.ceil((new Date(dateStr).getTime() - Date.now()) / 86400000);
  return diff;
}

function formatDate(dateStr: string | null) {
  if (!dateStr) return null;
  return new Date(dateStr).toLocaleDateString("en-IN", { day: "numeric", month: "short" });
}

function getAge(dateStr: string) {
  const today = new Date();
  const dob = new Date(dateStr);
  return today.getFullYear() - dob.getFullYear();
}

export default function CRMPage() {
  const { user } = useUser();
  const [leads, setLeads] = useState<Lead[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [activePage, setActivePage] = useState<ActivePage>("dashboard");

  // Lead detail modal
  const [selectedLead, setSelectedLead] = useState<Lead | null>(null);
  const [saving, setSaving] = useState(false);
  const [editPhone, setEditPhone] = useState("");
  const [editIncome, setEditIncome] = useState("");
  const [editFamilySize, setEditFamilySize] = useState("");
  const [editNotes, setEditNotes] = useState("");
  const [editDOB, setEditDOB] = useState("");
  const [editFollowUp, setEditFollowUp] = useState("");
  const [editStatus, setEditStatus] = useState<LeadStatus>("NEW");

  // Search/filter/sort for leads page
  const [search, setSearch] = useState("");
  const [filterStatus, setFilterStatus] = useState<LeadStatus | "ALL">("ALL");
  const [filterPolicy, setFilterPolicy] = useState("ALL");
  const [sortBy, setSortBy] = useState<"name" | "date" | "status">("date");

  // Birthday tab
  const [birthdayTab, setBirthdayTab] = useState<"today" | "week">("today");

  // Premium tab
  const [premiumTab, setPremiumTab] = useState<"urgent" | "soon">("urgent");

  useEffect(() => { fetchData(); }, []);

  const fetchData = async () => {
    try {
      const res = await fetch("/api/crm");
      const json = await res.json();
      if (json.success) {
        setLeads(json.data.leads);
        setStats(json.data.stats);
      }
    } catch { console.error("Failed to fetch"); }
    finally { setLoading(false); }
  };

  const openLead = (lead: Lead) => {
    setSelectedLead(lead);
    setEditPhone(lead.phone ?? "");
    setEditIncome(lead.income?.toString() ?? "");
    setEditFamilySize(lead.familySize?.toString() ?? "");
    setEditNotes(lead.notes ?? "");
    setEditDOB(lead.dateOfBirth ? lead.dateOfBirth.split("T")[0] : "");
    setEditFollowUp(lead.followUpAt ? lead.followUpAt.split("T")[0] : "");
    setEditStatus(lead.status);
  };

  const saveLead = async () => {
    if (!selectedLead) return;
    setSaving(true);
    try {
      await fetch("/api/crm", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          leadId: selectedLead.id,
          phone: editPhone, income: editIncome,
          familySize: editFamilySize, notes: editNotes,
          dateOfBirth: editDOB, followUpAt: editFollowUp,
          status: editStatus,
        }),
      });
      await fetchData();
      setSelectedLead(null);
    } catch { console.error("Failed to save"); }
    finally { setSaving(false); }
  };

  const markReminderDone = async (reminderId: string) => {
    await fetch("/api/reminders", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reminderId }),
    });
    await fetchData();
  };

  // Filtered/sorted leads
  const allPolicyTypes = Array.from(new Set(leads.flatMap(l => l.issuances.map(i => i.policyName))));

  const filteredLeads = leads
    .filter(l => filterStatus === "ALL" || l.status === filterStatus)
    .filter(l => filterPolicy === "ALL" || l.issuances.some(i => i.policyName === filterPolicy))
    .filter(l => l.householdName.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === "name") return a.householdName.localeCompare(b.householdName);
      if (sortBy === "status") return a.status.localeCompare(b.status);
      return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    });

  const birthdaysToday = leads.filter(l => l.isBirthdayToday);
  const birthdaysThisWeek = leads.filter(l => l.isBirthdayThisWeek && !l.isBirthdayToday);
  const premiumsUrgent = leads.filter(l => l.hasPremiumDueUrgent);
  const premiumsSoon = leads.filter(l => l.hasPremiumDue && !l.hasPremiumDueUrgent);

  const greeting = () => {
    const h = new Date().getHours();
    if (h < 12) return "Good morning";
    if (h < 17) return "Good afternoon";
    return "Good evening";
  };

  const navItems = [
    { id: "dashboard" as ActivePage, label: "Dashboard", icon: LayoutDashboard },
    { id: "birthdays" as ActivePage, label: "Birthday Reminders", icon: Gift, badge: stats?.birthdaysToday },
    { id: "premiums" as ActivePage, label: "Premium Due", icon: AlertTriangle, badge: stats?.premiumsDueCount },
    { id: "leads" as ActivePage, label: "All Leads", icon: Users },
  ];

  return (
    <div className="flex h-screen bg-slate-100 overflow-hidden">
      {/* Dark Sidebar */}
      <div className="w-64 bg-slate-900 flex flex-col shrink-0">
        {/* Logo */}
        <div className="p-5 border-b border-slate-800">
          <p className="text-xl font-bold text-white">Sure<span className="text-emerald-400">Im</span></p>
          <p className="text-xs text-slate-400 mt-0.5">Agent CRM Platform</p>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-3 space-y-1">
          <p className="text-xs text-slate-500 uppercase tracking-wider px-3 py-2">Main</p>
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setActivePage(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl transition-colors text-left ${
                activePage === item.id
                  ? "bg-emerald-600 text-white"
                  : "text-slate-400 hover:bg-slate-800 hover:text-white"
              }`}
            >
              <div className="flex items-center gap-3">
                <item.icon className="w-4 h-4" />
                <span className="text-sm font-medium">{item.label}</span>
              </div>
              {item.badge ? (
                <span className={`text-xs font-bold px-1.5 py-0.5 rounded-full min-w-[20px] text-center ${
                  activePage === item.id ? "bg-white/20 text-white" : "bg-red-500 text-white"
                }`}>
                  {item.badge}
                </span>
              ) : null}
            </button>
          ))}
        </nav>

        {/* Bottom links */}
        <div className="p-3 border-t border-slate-800 space-y-1">
          <Link href="/dashboard" className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-slate-400 hover:bg-slate-800 hover:text-white transition-colors">
            <MessageSquare className="w-4 h-4" />
            <span className="text-sm font-medium">Chat Interface</span>
          </Link>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <div className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between shrink-0">
          <div>
            <p className="font-semibold text-slate-900 capitalize">{activePage === "dashboard" ? "Dashboard" : activePage === "birthdays" ? "Birthday Reminders" : activePage === "premiums" ? "Premium Due Reminders" : "All Leads"}</p>
            <p className="text-xs text-slate-400">
              {activePage === "birthdays" && `${stats?.birthdaysToday ?? 0} customers celebrating today`}
              {activePage === "premiums" && `${stats?.premiumsDueCount ?? 0} policies due in next 30 days`}
              {activePage === "dashboard" && new Date().toLocaleDateString("en-IN", { weekday: "long", day: "numeric", month: "long", year: "numeric" })}
              {activePage === "leads" && `${filteredLeads.length} households`}
            </p>
          </div>
          <div className="flex items-center gap-2 bg-slate-100 rounded-xl px-3 py-2">
            <div className="w-7 h-7 rounded-full bg-emerald-600 flex items-center justify-center">
              <span className="text-white text-xs font-bold">{user?.firstName?.[0] ?? "A"}</span>
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-900">{user?.firstName ?? "Agent"}</p>
              <p className="text-xs text-slate-400">Field Agent</p>
            </div>
          </div>
        </div>

        {/* Page Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="w-6 h-6 text-emerald-500 animate-spin" />
            </div>
          ) : (
            <>
              {/* ─── DASHBOARD ─── */}
              {activePage === "dashboard" && stats && (
                <div className="space-y-6">
                  {/* Greeting Banner */}
                  <div className="bg-gradient-to-r from-emerald-600 to-emerald-500 rounded-2xl p-6 text-white">
                    <p className="text-xl font-bold">{greeting()}, {user?.firstName ?? "Agent"} 👋</p>
                    <p className="text-emerald-100 mt-1 text-sm">
                      You have {stats.birthdaysToday} birthday{stats.birthdaysToday !== 1 ? "s" : ""}, {stats.premiumsDueCount} premium due{stats.premiumsDueCount !== 1 ? "s" : ""}, and {stats.activeReminders} active reminder{stats.activeReminders !== 1 ? "s" : ""} today.
                    </p>
                  </div>

                  {/* Stats */}
                  <div className="grid grid-cols-4 gap-4">
                    {[
                      { label: "Total Households", value: stats.total, sub: "Across all villages", icon: Users, color: "text-blue-600", bg: "bg-blue-50" },
                      { label: "Birthdays Today", value: stats.birthdaysToday, sub: "Action needed", icon: Gift, color: "text-pink-600", bg: "bg-pink-50" },
                      { label: "Premiums Due", value: stats.premiumsDueCount, sub: "Next 30 days", icon: AlertTriangle, color: "text-yellow-600", bg: "bg-yellow-50" },
                      { label: "Policies Issued", value: stats.policiesIssued, sub: "Total active", icon: ShieldCheck, color: "text-emerald-600", bg: "bg-emerald-50" },
                    ].map(s => (
                      <div key={s.label} className="bg-white rounded-2xl p-4 border border-slate-100 shadow-sm">
                        <div className={`w-9 h-9 ${s.bg} rounded-xl flex items-center justify-center mb-3`}>
                          <s.icon className={`w-4 h-4 ${s.color}`} />
                        </div>
                        <p className="text-2xl font-bold text-slate-900">{s.value}</p>
                        <p className="text-xs text-slate-400 mt-0.5">{s.label}</p>
                        <p className="text-xs text-slate-300 mt-0.5">{s.sub}</p>
                      </div>
                    ))}
                  </div>

                  {/* Quick Actions */}
                  <div>
                    <p className="font-semibold text-slate-900 mb-3 flex items-center gap-2">
                      <Bell className="w-4 h-4 text-slate-400" /> Quick Actions Needed Today
                    </p>
                    <div className="space-y-3">
                      {birthdaysToday.map(l => (
                        <div key={l.id} className="bg-white rounded-2xl p-4 border border-slate-100 shadow-sm flex items-center justify-between">
                          <div>
                            <div className="flex items-center gap-2 mb-0.5">
                              <span className="text-base">🎂</span>
                              <p className="font-semibold text-slate-900">{l.householdName} turns {l.dateOfBirth ? getAge(l.dateOfBirth) : ""} today</p>
                              <span className="text-xs bg-pink-100 text-pink-600 px-2 py-0.5 rounded-full font-medium">Birthday</span>
                            </div>
                            <p className="text-xs text-slate-400 ml-6">{l.phone ?? "No phone"} · {l.issuances[0]?.policyName ?? "No policy"}</p>
                          </div>
                          <div className="flex gap-2">
                            <button onClick={() => openLead(l)} className="text-xs bg-emerald-600 hover:bg-emerald-700 text-white px-3 py-2 rounded-xl font-medium transition-colors">Send Wish</button>
                            {l.phone && <a href={`tel:${l.phone}`} className="text-xs border border-slate-200 hover:bg-slate-50 text-slate-700 px-3 py-2 rounded-xl font-medium transition-colors flex items-center gap-1"><Phone className="w-3 h-3" /> Call</a>}
                          </div>
                        </div>
                      ))}

                      {premiumsUrgent.slice(0, 3).map(l => (
                        <div key={l.id} className="bg-white rounded-2xl p-4 border border-slate-100 shadow-sm flex items-center justify-between">
                          <div>
                            <div className="flex items-center gap-2 mb-0.5">
                              <span className="text-base">⚠️</span>
                              <p className="font-semibold text-slate-900">{l.householdName} — Premium Due</p>
                              <span className="text-xs bg-red-100 text-red-600 px-2 py-0.5 rounded-full font-medium">URGENT</span>
                            </div>
                            <p className="text-xs text-slate-400 ml-6">
                              {l.premiumsDue[0]?.policyName} · {l.premiumsDue[0]?.expiresAt ? `${daysUntil(l.premiumsDue[0].expiresAt)} days left` : ""}
                            </p>
                          </div>
                          <div className="flex gap-2">
                            {l.phone && <a href={`tel:${l.phone}`} className="text-xs bg-red-500 hover:bg-red-600 text-white px-3 py-2 rounded-xl font-medium transition-colors flex items-center gap-1"><Phone className="w-3 h-3" /> Call Now</a>}
                            <button onClick={() => openLead(l)} className="text-xs border border-slate-200 hover:bg-slate-50 text-slate-700 px-3 py-2 rounded-xl font-medium">Remind</button>
                          </div>
                        </div>
                      ))}

                      {leads.flatMap(l => l.reminders.map(r => ({ ...r, lead: l }))).slice(0, 2).map(r => (
                        <div key={r.id} className="bg-white rounded-2xl p-4 border border-slate-100 shadow-sm flex items-center justify-between">
                          <div>
                            <div className="flex items-center gap-2 mb-0.5">
                              <span className="text-base">📋</span>
                              <p className="font-semibold text-slate-900">{r.lead.householdName} — {r.type === "BIRTHDAY" ? "Birthday" : "Follow-up"}</p>
                            </div>
                            <p className="text-xs text-slate-400 ml-6">{r.note ?? "No note"} · {formatDate(r.scheduledAt)}</p>
                          </div>
                          <button onClick={() => markReminderDone(r.id)} className="text-xs border border-slate-200 hover:bg-slate-50 text-slate-700 px-3 py-2 rounded-xl font-medium flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3" /> Done
                          </button>
                        </div>
                      ))}

                      {birthdaysToday.length === 0 && premiumsUrgent.length === 0 && (
                        <div className="bg-white rounded-2xl p-8 text-center border border-slate-100">
                          <p className="text-2xl mb-2">🎉</p>
                          <p className="text-slate-500 text-sm">No urgent actions today!</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* ─── BIRTHDAYS ─── */}
              {activePage === "birthdays" && <BirthdaySection />}
              {/* ─── PREMIUMS ─── */}
              {activePage === "premiums" && (
                <div className="space-y-4">
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { label: "Due This Week", value: stats?.premiumsDueUrgentCount ?? 0, sub: "Urgent action", color: "text-red-600" },
                      { label: "Due This Month", value: stats?.premiumsDueCount ?? 0, sub: "Monitor closely", color: "text-yellow-600" },
                      { label: "Avg Renewal Score", value: "—", sub: "Across all due", color: "text-emerald-600" },
                    ].map(s => (
                      <div key={s.label} className="bg-white rounded-2xl p-4 border border-slate-100 shadow-sm">
                        <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
                        <p className="text-sm font-medium text-slate-700 mt-1">{s.label}</p>
                        <p className="text-xs text-slate-400">{s.sub}</p>
                      </div>
                    ))}
                  </div>

                  {/* Tabs */}
                  <div className="flex gap-2">
                    {[{ key: "urgent", label: `🔴 Urgent (${premiumsUrgent.length})` }, { key: "soon", label: `🟡 Soon (${premiumsSoon.length})` }].map(t => (
                      <button key={t.key} onClick={() => setPremiumTab(t.key as "urgent" | "soon")}
                        className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${premiumTab === t.key ? "bg-slate-900 text-white" : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-50"}`}>
                        {t.label}
                      </button>
                    ))}
                  </div>

                  <div className="space-y-3">
                    {(premiumTab === "urgent" ? premiumsUrgent : premiumsSoon).map(l => (
                      <div key={l.id} className="bg-white rounded-2xl p-5 border border-slate-100 shadow-sm">
                        <div className="flex items-start justify-between mb-1">
                          <p className="font-semibold text-slate-900 text-base">{l.householdName}</p>
                          {l.premiumsDue[0]?.expiresAt && (
                            <span className={`text-xs font-semibold px-2 py-1 rounded-full ${daysUntil(l.premiumsDue[0].expiresAt) <= 7 ? "bg-red-100 text-red-600" : "bg-yellow-100 text-yellow-600"}`}>
                              Due in {daysUntil(l.premiumsDue[0].expiresAt)} days
                            </span>
                          )}
                        </div>
                        {l.notes && <p className="text-xs text-slate-400 mb-3">{l.notes.split(",")[0]}</p>}
                        <div className="flex gap-3 text-xs text-slate-500 mb-3 flex-wrap">
                          {l.premiumsDue[0] && <span className="bg-slate-50 px-2 py-1 rounded-lg">📋 {l.premiumsDue[0].policyName}</span>}
                          {l.premiumsDue[0]?.premiumAmount && <span className="bg-slate-50 px-2 py-1 rounded-lg">💰 ₹{l.premiumsDue[0].premiumAmount}</span>}
                          {l.phone && <span className="bg-slate-50 px-2 py-1 rounded-lg flex items-center gap-1"><Phone className="w-3 h-3" />{l.phone}</span>}
                        </div>
                        <div className="flex gap-2">
                          {l.phone && (
                            <a href={`tel:${l.phone}`} className="flex-1 bg-red-500 hover:bg-red-600 text-white text-sm font-medium py-2.5 rounded-xl transition-colors flex items-center justify-center gap-2">
                              <Phone className="w-4 h-4" /> Call Now
                            </a>
                          )}
                          <button onClick={() => openLead(l)} className="flex-1 border border-slate-200 hover:bg-slate-50 text-slate-700 text-sm font-medium py-2.5 rounded-xl transition-colors">
                            💬 Send Reminder
                          </button>
                          <button onClick={() => openLead(l)} className="flex-1 border border-slate-200 hover:bg-slate-50 text-slate-700 text-sm font-medium py-2.5 rounded-xl transition-colors flex items-center justify-center gap-1">
                            <Calendar className="w-3.5 h-3.5" /> Visit
                          </button>
                        </div>
                      </div>
                    ))}
                    {(premiumTab === "urgent" ? premiumsUrgent : premiumsSoon).length === 0 && (
                      <div className="bg-white rounded-2xl p-10 text-center border border-slate-100">
                        <p className="text-3xl mb-2">✅</p>
                        <p className="text-slate-500">No {premiumTab === "urgent" ? "urgent" : "upcoming"} premiums due</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* ─── ALL LEADS ─── */}
              {activePage === "leads" && (
                <div className="space-y-4">
                  {/* Search + Filter + Sort */}
                  <div className="flex gap-3 flex-wrap">
                    <div className="flex-1 min-w-48 flex items-center gap-2 bg-white rounded-xl px-3 py-2.5 border border-slate-200">
                      <Search className="w-4 h-4 text-slate-400 shrink-0" />
                      <input
                        type="text"
                        placeholder="Search households..."
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                        className="bg-transparent text-sm text-slate-700 placeholder:text-slate-400 outline-none w-full"
                      />
                      {search && <button onClick={() => setSearch("")}><X className="w-3.5 h-3.5 text-slate-400" /></button>}
                    </div>

                    <select
                      value={filterStatus}
                      onChange={e => setFilterStatus(e.target.value as LeadStatus | "ALL")}
                      className="bg-white border border-slate-200 rounded-xl px-3 py-2.5 text-sm text-slate-700 outline-none"
                    >
                      <option value="ALL">All Statuses</option>
                      <option value="NEW">New</option>
                      <option value="CONTACTED">Contacted</option>
                      <option value="POLICY_ISSUED">Policy Issued</option>
                      <option value="REJECTED">Rejected</option>
                    </select>

                    <select
                      value={filterPolicy}
                      onChange={e => setFilterPolicy(e.target.value)}
                      className="bg-white border border-slate-200 rounded-xl px-3 py-2.5 text-sm text-slate-700 outline-none"
                    >
                      <option value="ALL">All Policies</option>
                      {allPolicyTypes.map(p => <option key={p} value={p}>{p}</option>)}
                    </select>

                    <select
                      value={sortBy}
                      onChange={e => setSortBy(e.target.value as "name" | "date" | "status")}
                      className="bg-white border border-slate-200 rounded-xl px-3 py-2.5 text-sm text-slate-700 outline-none flex items-center gap-2"
                    >
                      <option value="date">Sort: Recent</option>
                      <option value="name">Sort: Name</option>
                      <option value="status">Sort: Status</option>
                    </select>
                  </div>

                  <p className="text-sm text-slate-500">{filteredLeads.length} households found</p>

                  <div className="space-y-2">
                    {filteredLeads.map(l => (
                      <button key={l.id} onClick={() => openLead(l)}
                        className="w-full bg-white rounded-2xl p-4 border border-slate-100 shadow-sm hover:shadow-md hover:border-emerald-200 transition-all text-left"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-emerald-100 flex items-center justify-center shrink-0">
                              <span className="text-emerald-700 font-bold">{l.householdName[0]}</span>
                            </div>
                            <div>
                              <div className="flex items-center gap-2">
                                <p className="font-semibold text-slate-900">{l.householdName}</p>
                                {l.isBirthdayToday && <span>🎂</span>}
                                {l.hasPremiumDueUrgent && <span className="text-xs bg-red-100 text-red-600 px-1.5 py-0.5 rounded-full">Premium Due</span>}
                              </div>
                              <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                                {l.phone && <span className="text-xs text-slate-400">{l.phone}</span>}
                                {l.familySize && <span className="text-xs text-slate-400">· Family of {l.familySize}</span>}
                                {l.income && <span className="text-xs text-slate-400">· ₹{l.income.toLocaleString()}/mo</span>}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            {l.issuances.length > 0 && (
                              <span className="text-xs bg-emerald-50 text-emerald-600 px-2 py-0.5 rounded-full">
                                {l.issuances.length} policy
                              </span>
                            )}
                            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_COLORS[l.status]}`}>
                              {l.status.replace("_", " ")}
                            </span>
                          </div>
                        </div>
                        {l.issuances.length > 0 && (
                          <div className="flex gap-1.5 mt-2 ml-13 flex-wrap">
                            {l.issuances.map(i => (
                              <span key={i.id} className="text-xs bg-slate-50 text-slate-500 px-2 py-0.5 rounded-lg">{i.policyName}</span>
                            ))}
                          </div>
                        )}
                      </button>
                    ))}
                    {filteredLeads.length === 0 && (
                      <div className="bg-white rounded-2xl p-10 text-center border border-slate-100">
                        <p className="text-3xl mb-2">🔍</p>
                        <p className="text-slate-500">No households match your filters</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Lead Detail Modal */}
      {selectedLead && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setSelectedLead(null)} />
          <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 max-h-[90vh] overflow-y-auto">
            <div className="sticky top-0 bg-white border-b border-slate-100 px-5 py-4 flex items-center justify-between rounded-t-2xl">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-full bg-emerald-100 flex items-center justify-center">
                  <span className="text-emerald-700 font-bold">{selectedLead.householdName[0]}</span>
                </div>
                <p className="font-semibold text-slate-900">{selectedLead.householdName}</p>
              </div>
              <button onClick={() => setSelectedLead(null)} className="w-7 h-7 rounded-full bg-slate-100 hover:bg-slate-200 flex items-center justify-center">
                <X className="w-4 h-4 text-slate-500" />
              </button>
            </div>

            <div className="px-5 py-4 space-y-4">
              {/* Status */}
              <div className="flex gap-2 flex-wrap">
                {(["NEW", "CONTACTED", "POLICY_ISSUED", "REJECTED"] as LeadStatus[]).map(s => (
                  <button key={s} onClick={() => setEditStatus(s)}
                    className={`text-xs font-medium px-3 py-1.5 rounded-xl border transition-colors ${editStatus === s ? STATUS_COLORS[s] + " border-current" : "bg-slate-50 text-slate-500 border-slate-200"}`}>
                    {s.replace("_", " ")}
                  </button>
                ))}
              </div>

              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Phone", value: editPhone, setter: setEditPhone, type: "tel", placeholder: "9845000000" },
                  { label: "Family Size", value: editFamilySize, setter: setEditFamilySize, type: "number", placeholder: "4" },
                  { label: "Monthly Income (₹)", value: editIncome, setter: setEditIncome, type: "number", placeholder: "8000" },
                  { label: "Date of Birth 🎂", value: editDOB, setter: setEditDOB, type: "date", placeholder: "" },
                ].map(f => (
                  <div key={f.label}>
                    <label className="text-xs text-slate-400 mb-1 block">{f.label}</label>
                    <input type={f.type} value={f.value} onChange={e => f.setter(e.target.value)} placeholder={f.placeholder}
                      className="w-full bg-slate-50 rounded-xl px-3 py-2 text-sm text-slate-700 outline-none focus:ring-2 focus:ring-emerald-500" />
                  </div>
                ))}
              </div>

              <div>
                <label className="text-xs text-slate-400 mb-1 block">Follow-up Date</label>
                <input type="date" value={editFollowUp} onChange={e => setEditFollowUp(e.target.value)}
                  className="w-full bg-slate-50 rounded-xl px-3 py-2 text-sm text-slate-700 outline-none focus:ring-2 focus:ring-emerald-500" />
              </div>

              <div>
                <label className="text-xs text-slate-400 mb-1 block">Notes</label>
                <textarea value={editNotes} onChange={e => setEditNotes(e.target.value)} rows={2}
                  className="w-full bg-slate-50 rounded-xl px-3 py-2 text-sm text-slate-700 outline-none focus:ring-2 focus:ring-emerald-500 resize-none" />
              </div>

              {selectedLead.issuances.length > 0 && (
                <div>
                  <label className="text-xs text-slate-400 mb-1.5 block">Issued Policies</label>
                  {selectedLead.issuances.map(i => (
                    <div key={i.id} className="flex items-center justify-between bg-emerald-50 rounded-xl px-3 py-2 mb-1.5">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                        <p className="text-sm font-medium text-slate-900">{i.policyName}</p>
                      </div>
                      {i.premiumAmount && <p className="text-xs text-slate-500">₹{i.premiumAmount}/yr</p>}
                    </div>
                  ))}
                </div>
              )}

              {selectedLead.reminders.length > 0 && (
                <div>
                  <label className="text-xs text-slate-400 mb-1.5 block">Active Reminders</label>
                  {selectedLead.reminders.map(r => (
                    <div key={r.id} className="flex items-center justify-between bg-blue-50 rounded-xl px-3 py-2 mb-1.5">
                      <div>
                        <p className="text-xs font-semibold text-slate-900">{r.type === "BIRTHDAY" ? "🎂 Birthday" : "📞 Follow-up"}</p>
                        <p className="text-xs text-slate-500">{formatDate(r.scheduledAt)}{r.note ? ` · ${r.note}` : ""}</p>
                      </div>
                      <button onClick={() => markReminderDone(r.id)} className="w-6 h-6 rounded-full border-2 border-blue-300 hover:bg-blue-100 flex items-center justify-center">
                        <CheckCircle2 className="w-3.5 h-3.5 text-blue-400" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="sticky bottom-0 bg-white border-t border-slate-100 px-5 py-4 rounded-b-2xl">
              <button onClick={saveLead} disabled={saving}
                className="w-full bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-200 text-white text-sm font-medium py-3 rounded-xl transition-colors flex items-center justify-center gap-2">
                {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle2 className="w-4 h-4" />}
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}