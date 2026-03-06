"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Sparkles, ShieldCheck, Loader2, CheckCircle2 } from "lucide-react";

interface Message {
  id: string;
  role: "agent" | "ai";
  content: string;
  policies?: Policy[];
  timestamp: Date;
}

interface Policy {
  name: string;
  provider: string;
  premium: string;
  coverage: string;
  tag: string;
}

interface Lead {
  id: string;
  householdName: string;
  lastMessage: string;
  time: string;
  unread: number;
}

interface DbMessage {
  id: string;
  role: string;
  content: string;
  policies: Policy[] | null;
  createdAt: string;
}

interface ChatAreaProps {
  lead: Lead;
}

const WELCOME_MESSAGE = (name: string): Message => ({
  id: "welcome",
  role: "ai",
  content: `Hello! I'm your AI insurance assistant. Tell me about ${name}'s household — their income, family size, occupation, and any specific needs. I'll recommend the best policies for them.`,
  timestamp: new Date(),
});

export default function ChatArea({ lead }: ChatAreaProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [issuingPolicy, setIssuingPolicy] = useState<string | null>(null);
  const [issuedPolicies, setIssuedPolicies] = useState<Set<string>>(new Set());
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMessages([]);
    setInput("");
    setLoadingHistory(true);
    setIssuedPolicies(new Set());
    loadAll();
  }, [lead.id]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load messages and issued policies in parallel
  const loadAll = async () => {
    try {
      const [messagesRes, issuancesRes] = await Promise.all([
        fetch(`/api/messages?leadId=${lead.id}`),
        fetch(`/api/issuances?leadId=${lead.id}`),
      ]);

      const [messagesJson, issuancesJson] = await Promise.all([
        messagesRes.json(),
        issuancesRes.json(),
      ]);

      // Load messages
      if (messagesJson.success && messagesJson.data.length > 0) {
        const mapped: Message[] = messagesJson.data.map((m: DbMessage) => ({
          id: m.id,
          role: m.role === "AGENT" ? "agent" : "ai",
          content: m.content,
          policies: m.policies ?? undefined,
          timestamp: new Date(m.createdAt),
        }));
        setMessages(mapped);
      } else {
        setMessages([WELCOME_MESSAGE(lead.householdName)]);
      }

      // Load issued policies into Set
      if (issuancesJson.success && issuancesJson.data.length > 0) {
        const names = new Set<string>(
          issuancesJson.data.map((i: { policyName: string }) => i.policyName)
        );
        setIssuedPolicies(names);
      }
    } catch {
      setMessages([WELCOME_MESSAGE(lead.householdName)]);
    } finally {
      setLoadingHistory(false);
    }
  };

  const saveMessage = async (role: "agent" | "ai", content: string, policies?: Policy[]) => {
    try {
      await fetch("/api/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ leadId: lead.id, role, content, policies: policies ?? null }),
      });
    } catch {
      console.error("Failed to save message");
    }
  };

  const issuePolicy = async (policy: Policy) => {
    if (issuingPolicy || issuedPolicies.has(policy.name)) return;
    setIssuingPolicy(policy.name);
    try {
      const res = await fetch("/api/issuances", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          leadId: lead.id,
          policyName: policy.name,
          policyProvider: policy.provider,
          premiumAmount: policy.premium,
        }),
      });
      const json = await res.json();
      if (json.success) {
        setIssuedPolicies((prev) => new Set(prev).add(policy.name));
      }
    } catch {
      console.error("Failed to issue policy");
    } finally {
      setIssuingPolicy(null);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "agent",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const sentInput = input;
    setInput("");
    setLoading(true);

    await saveMessage("agent", sentInput);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: sentInput, householdName: lead.householdName }),
      });

      const json = await res.json();

      if (json.success) {
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "ai",
          content: json.data.analysis,
          policies: json.data.policies,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, aiMessage]);
        await saveMessage("ai", json.data.analysis, json.data.policies);
      } else {
        throw new Error("AI error");
      }
    } catch {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "ai",
        content: "Sorry, I couldn't process that. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (date: Date) =>
    date.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });

  const messageCount = messages.filter((m) => m.role === "agent").length;

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 px-6 py-4 flex items-center gap-3">
        <div className="w-9 h-9 rounded-full bg-emerald-100 flex items-center justify-center">
          <span className="text-emerald-700 font-semibold text-sm">{lead.householdName[0]}</span>
        </div>
        <div>
          <p className="font-semibold text-slate-900 text-sm">{lead.householdName}</p>
          <p className="text-xs text-slate-400">
            {loadingHistory
              ? "Loading..."
              : messageCount === 0
              ? "New conversation"
              : `${messageCount} message${messageCount === 1 ? "" : "s"}`}
          </p>
        </div>
        {issuedPolicies.size > 0 && (
          <div className="ml-auto flex items-center gap-1.5 bg-emerald-50 text-emerald-700 text-xs font-medium px-3 py-1 rounded-full">
            <CheckCircle2 className="w-3 h-3" />
            {issuedPolicies.size} policy issued
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {loadingHistory ? (
          <div className="flex flex-col gap-3 animate-pulse">
            <div className="flex justify-start">
              <div className="h-16 w-72 bg-slate-100 rounded-2xl" />
            </div>
            <div className="flex justify-end">
              <div className="h-10 w-48 bg-emerald-50 rounded-2xl" />
            </div>
            <div className="flex justify-start">
              <div className="h-24 w-80 bg-slate-100 rounded-2xl" />
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === "agent" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-lg ${
                  msg.role === "agent" ? "items-end" : "items-start"
                } flex flex-col gap-1`}
              >
                {msg.role === "ai" && (
                  <div className="flex items-center gap-2 mb-1">
                    <div className="w-6 h-6 bg-emerald-600 rounded-full flex items-center justify-center">
                      <Sparkles className="w-3 h-3 text-white" />
                    </div>
                    <span className="text-xs text-slate-400">AI Assistant</span>
                  </div>
                )}
                <div
                  className={`px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                    msg.role === "agent"
                      ? "bg-emerald-600 text-white rounded-tr-sm"
                      : "bg-white border border-slate-100 text-slate-700 rounded-tl-sm shadow-sm"
                  }`}
                >
                  {msg.content}
                </div>

                {msg.policies && msg.policies.length > 0 && (
                  <div className="mt-2 space-y-2 w-full">
                    {msg.policies.map((policy) => {
                      const issued = issuedPolicies.has(policy.name);
                      const issuing = issuingPolicy === policy.name;
                      return (
                        <div
                          key={policy.name}
                          className={`bg-white border rounded-2xl p-4 shadow-sm transition-all ${
                            issued
                              ? "border-emerald-300 bg-emerald-50/50"
                              : "border-slate-100 hover:border-emerald-200"
                          }`}
                        >
                          <div className="flex items-start justify-between mb-2">
                            <div>
                              <p className="font-semibold text-slate-900 text-sm">{policy.name}</p>
                              <p className="text-xs text-slate-400">{policy.provider}</p>
                            </div>
                            <span className="text-xs bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-full font-medium">
                              {policy.tag}
                            </span>
                          </div>
                          <div className="flex items-center justify-between mt-3">
                            <div>
                              <p className="text-xs text-slate-400">Premium</p>
                              <p className="text-sm font-semibold text-slate-900">{policy.premium}</p>
                            </div>
                            <div>
                              <p className="text-xs text-slate-400">Coverage</p>
                              <p className="text-sm font-semibold text-slate-900">{policy.coverage}</p>
                            </div>
                            <button
                              onClick={() => issuePolicy(policy)}
                              disabled={issued || !!issuing}
                              className={`flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-xl transition-colors ${
                                issued
                                  ? "bg-emerald-100 text-emerald-700 cursor-default"
                                  : "bg-emerald-600 hover:bg-emerald-700 text-white"
                              }`}
                            >
                              {issuing ? (
                                <Loader2 className="w-3 h-3 animate-spin" />
                              ) : issued ? (
                                <CheckCircle2 className="w-3 h-3" />
                              ) : (
                                <ShieldCheck className="w-3 h-3" />
                              )}
                              {issued ? "Issued" : "Issue Policy"}
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}

                <p className="text-xs text-slate-300 px-1">{formatTime(msg.timestamp)}</p>
              </div>
            </div>
          ))
        )}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-slate-100 rounded-2xl px-4 py-3 shadow-sm">
              <div className="flex gap-1 items-center">
                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="bg-white border-t border-slate-200 px-6 py-4">
        <div className="flex items-end gap-3">
          <div className="flex-1 bg-slate-50 rounded-2xl px-4 py-3">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder={`Describe ${lead.householdName}'s household...`}
              className="w-full bg-transparent text-sm text-slate-700 placeholder:text-slate-400 outline-none resize-none max-h-32"
              rows={1}
            />
          </div>
          <button
            onClick={sendMessage}
            disabled={!input.trim() || loading}
            className="w-10 h-10 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-200 rounded-full flex items-center justify-center transition-colors shrink-0"
          >
            <Send className="w-4 h-4 text-white" />
          </button>
        </div>
        <p className="text-xs text-slate-300 mt-2 text-center">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}