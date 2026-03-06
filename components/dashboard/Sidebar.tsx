"use client";

import { useState, useEffect, useRef } from "react";
import { UserButton } from "@clerk/nextjs";
import { MessageSquare, Plus, Search, X, Loader2, Trash2, MoreVertical } from "lucide-react";
import Link from "next/link";

interface Lead {
  id: string;
  householdName: string;
  lastMessage: string;
  time: string;
  unread: number;
}

interface SidebarProps {
  user: { name: string; avatar: string };
  onSelectLead?: (lead: Lead | null) => void;
  activeLead?: Lead | null;
}

export default function Sidebar({ user, onSelectLead, activeLead }: SidebarProps) {
  const [search, setSearch] = useState("");
  const [leads, setLeads] = useState<Lead[]>([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [householdName, setHouseholdName] = useState("");
  const [creating, setCreating] = useState(false);
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchLeads();
  }, []);

  // Close menu on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpenId(null);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const fetchLeads = async () => {
    try {
      const res = await fetch("/api/leads");
      const json = await res.json();
      if (json.success) {
        const mapped = json.data.map((l: any) => ({
          id: l.id,
          householdName: l.householdName,
          lastMessage: l.notes ?? "New conversation",
          time: new Date(l.createdAt).toLocaleTimeString("en-IN", {
            hour: "2-digit",
            minute: "2-digit",
          }),
          unread: 0,
        }));
        setLeads(mapped);
      }
    } catch {
      console.error("Failed to fetch leads");
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    if (!householdName.trim() || creating) return;
    setCreating(true);
    try {
      const res = await fetch("/api/leads", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ householdName: householdName.trim() }),
      });
      const json = await res.json();
      if (json.success) {
        const newLead: Lead = {
          id: json.data.id,
          householdName: json.data.householdName,
          lastMessage: "New conversation",
          time: "Now",
          unread: 0,
        };
        setLeads((prev) => [newLead, ...prev]);
        onSelectLead?.(newLead);
      }
    } catch {
      console.error("Failed to create lead");
    } finally {
      setHouseholdName("");
      setShowModal(false);
      setCreating(false);
    }
  };

  const handleDelete = async (lead: Lead) => {
    setDeletingId(lead.id);
    setMenuOpenId(null);
    try {
      const res = await fetch(`/api/leads/${lead.id}`, { method: "DELETE" });
      const json = await res.json();
      if (json.success) {
        setLeads((prev) => prev.filter((l) => l.id !== lead.id));
        if (activeLead?.id === lead.id) onSelectLead?.(null);
      }
    } catch {
      console.error("Failed to delete lead");
    } finally {
      setDeletingId(null);
    }
  };

  const filtered = leads.filter((l) =>
    l.householdName.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <>
      <div className="w-80 flex flex-col border-r border-slate-200 bg-white">
        {/* Header */}
        <Link href="/crm">CRM</Link>
        <div className="p-4 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <UserButton afterSignOutUrl="/" />
            <div>
              <p className="text-sm font-semibold text-slate-900">{user.name}</p>
              <p className="text-xs text-slate-400">Agent</p>
            </div>
          </div>
          <button
            onClick={() => setShowModal(true)}
            className="w-8 h-8 bg-emerald-600 hover:bg-emerald-700 rounded-full flex items-center justify-center transition-colors"
          >
            <Plus className="w-4 h-4 text-white" />
          </button>
        </div>

        {/* Search */}
        <div className="p-3 border-b border-slate-100">
          <div className="flex items-center gap-2 bg-slate-50 rounded-xl px-3 py-2">
            <Search className="w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search households..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="bg-transparent text-sm text-slate-700 placeholder:text-slate-400 outline-none w-full"
            />
          </div>
        </div>

        {/* Lead list */}
        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="w-5 h-5 text-emerald-400 animate-spin" />
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center p-6">
              <MessageSquare className="w-8 h-8 text-slate-200 mb-3" />
              <p className="text-sm text-slate-400">No households yet</p>
              <p className="text-xs text-slate-300 mt-1">Click + to add one</p>
            </div>
          ) : (
            filtered.map((lead) => (
              <div
                key={lead.id}
                className={`relative flex items-start group border-b border-slate-50 ${
                  activeLead?.id === lead.id ? "bg-emerald-50 border-l-2 border-l-emerald-600" : "hover:bg-slate-50"
                }`}
              >
                {/* Lead button */}
                <button
                  onClick={() => onSelectLead?.(lead)}
                  className="flex-1 p-4 flex items-start gap-3 text-left min-w-0"
                >
                  <div className="w-10 h-10 rounded-full bg-emerald-100 flex items-center justify-center shrink-0">
                    {deletingId === lead.id ? (
                      <Loader2 className="w-4 h-4 text-emerald-600 animate-spin" />
                    ) : (
                      <span className="text-emerald-700 font-semibold text-sm">
                        {lead.householdName[0]}
                      </span>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-0.5">
                      <p className="text-sm font-semibold text-slate-900 truncate">
                        {lead.householdName}
                      </p>
                      <p className="text-xs text-slate-400 shrink-0 ml-2">{lead.time}</p>
                    </div>
                    <p className="text-xs text-slate-400 truncate">{lead.lastMessage}</p>
                  </div>
                </button>

                {/* 3-dot menu */}
                <div className="relative flex items-center pr-2 pt-4" ref={menuOpenId === lead.id ? menuRef : null}>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setMenuOpenId(menuOpenId === lead.id ? null : lead.id);
                    }}
                    className="w-6 h-6 rounded-full hover:bg-slate-200 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <MoreVertical className="w-3.5 h-3.5 text-slate-400" />
                  </button>

                  {menuOpenId === lead.id && (
                    <div className="absolute right-0 top-8 z-20 bg-white border border-slate-100 rounded-xl shadow-lg py-1 w-36">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(lead);
                        }}
                        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-500 hover:bg-red-50 transition-colors"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                        Delete
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Bottom branding */}
        <div className="p-4 border-t border-slate-100">
          <p className="text-xs text-slate-300 text-center">
            Sure<span className="text-emerald-400">Im</span> · Agent Platform
          </p>
        </div>
      </div>

      {/* New Conversation Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/30 backdrop-blur-sm" onClick={() => setShowModal(false)} />
          <div className="relative bg-white rounded-2xl shadow-xl w-full max-w-sm mx-4 p-6">
            <button
              onClick={() => setShowModal(false)}
              className="absolute top-4 right-4 w-7 h-7 rounded-full bg-slate-100 hover:bg-slate-200 flex items-center justify-center transition-colors"
            >
              <X className="w-4 h-4 text-slate-500" />
            </button>
            <div className="w-10 h-10 bg-emerald-100 rounded-full flex items-center justify-center mb-4">
              <Plus className="w-5 h-5 text-emerald-600" />
            </div>
            <h2 className="text-base font-semibold text-slate-900 mb-1">New Household</h2>
            <p className="text-sm text-slate-400 mb-5">Enter the household name to start a new conversation.</p>
            <input
              type="text"
              autoFocus
              placeholder="e.g. Ramesh Kumar"
              value={householdName}
              onChange={(e) => setHouseholdName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
              className="w-full bg-slate-50 rounded-xl px-4 py-3 text-sm text-slate-700 placeholder:text-slate-400 outline-none focus:ring-2 focus:ring-emerald-500 mb-4"
            />
            <button
              onClick={handleCreate}
              disabled={!householdName.trim() || creating}
              className="w-full bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-200 disabled:text-slate-400 text-white text-sm font-medium py-3 rounded-xl transition-colors flex items-center justify-center gap-2"
            >
              {creating && <Loader2 className="w-4 h-4 animate-spin" />}
              Start Conversation
            </button>
          </div>
        </div>
      )}
    </>
  );
}