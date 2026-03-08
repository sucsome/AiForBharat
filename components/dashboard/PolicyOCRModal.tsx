// components/dashboard/PolicyOCRModal.tsx
"use client";

import { useRef, useState } from "react";
import { X, Upload, CheckCircle2, XCircle, Loader2, ShieldCheck } from "lucide-react";

interface Policy {
  name: string;
  provider: string;
  premium: string;
  coverage: string;
  tag: string;
}

interface Props {
  policy: Policy;
  householdName: string;
  onSuccess: (policy: Policy) => void;
  onClose: () => void;
}

type DocStatus = "pending" | "uploading" | "valid" | "invalid";

interface DocState {
  docType: "Aadhaar Card" | "PAN Card";
  status: DocStatus;
  preview: string | null;
  extractedName: string;
  reason: string;
}

const INITIAL_DOCS: DocState[] = [
  { docType: "Aadhaar Card", status: "pending", preview: null, extractedName: "", reason: "" },
  { docType: "PAN Card",     status: "pending", preview: null, extractedName: "", reason: "" },
];

// Status color map
const statusColor = {
  pending:   { border: "rgba(0,0,0,0.08)", headerBg: "#f8faf9", icon: "#cbd5e1" },
  uploading: { border: "rgba(5,150,105,0.3)", headerBg: "rgba(5,150,105,0.06)", icon: "#059669" },
  valid:     { border: "rgba(5,150,105,0.35)", headerBg: "rgba(5,150,105,0.07)", icon: "#059669" },
  invalid:   { border: "rgba(239,68,68,0.35)", headerBg: "rgba(239,68,68,0.06)", icon: "#ef4444" },
};

function StatusIcon({ status }: { status: DocStatus }) {
  if (status === "uploading") return <Loader2 size={13} color="#059669" className="animate-spin" />;
  if (status === "valid")     return <CheckCircle2 size={13} color="#059669" />;
  if (status === "invalid")   return <XCircle size={13} color="#ef4444" />;
  return (
    <div style={{
      width: 13, height: 13, borderRadius: "50%",
      border: "2px solid #cbd5e1",
    }} />
  );
}

export default function PolicyOCRModal({ policy, householdName, onSuccess, onClose }: Props) {
  const [docs, setDocs] = useState<DocState[]>(INITIAL_DOCS);
  const [issuing, setIssuing] = useState(false);
  const fileInputRefs = useRef<(HTMLInputElement | null)[]>([]);

  const fileToBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve((reader.result as string).split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

  const handleFileSelect = async (idx: number, file: File) => {
    const preview = URL.createObjectURL(file);
    setDocs((prev) =>
      prev.map((d, i) =>
        i === idx ? { ...d, preview, status: "uploading", extractedName: "", reason: "" } : d
      )
    );

    try {
      const base64 = await fileToBase64(file);
      const res = await fetch("/api/ocr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          imageBase64: base64,
          mediaType: file.type || "image/jpeg",
          expectedDocType: docs[idx].docType,
          householdName,
        }),
      });
      const json = await res.json();
      if (!json.success) throw new Error(json.error);

      setDocs((prev) =>
        prev.map((d, i) =>
          i === idx ? {
            ...d,
            status: json.valid ? "valid" : "invalid",
            extractedName: json.extractedName ?? "",
            reason: json.reason ?? "",
          } : d
        )
      );
    } catch {
      setDocs((prev) =>
        prev.map((d, i) =>
          i === idx ? { ...d, status: "invalid", reason: "Validation failed. Please try again." } : d
        )
      );
    }
  };

  const allValid     = docs.every((d) => d.status === "valid");
  const validCount   = docs.filter((d) => d.status === "valid").length;
  const anyUploading = docs.some((d) => d.status === "uploading");

  const handleIssue = async () => {
    if (!allValid || issuing) return;
    setIssuing(true);
    await onSuccess(policy);
    setIssuing(false);
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');
        .upload-zone:hover { border-color: rgba(5,150,105,0.5) !important; background: rgba(5,150,105,0.03) !important; }
        .upload-zone:hover .upload-icon { color: #059669 !important; }
        .issue-btn:hover:not(:disabled) { background: #047857 !important; transform: translateY(-1px); }
        .retry-btn:hover { text-decoration: underline; }
      `}</style>

      {/* Backdrop */}
      <div
        onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
        style={{
          position: "fixed", inset: 0, zIndex: 50,
          display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(12,26,18,0.5)",
          backdropFilter: "blur(8px)",
          padding: 16,
          fontFamily: "'DM Sans', sans-serif",
        }}
      >
        <div
          onClick={(e) => e.stopPropagation()}
          style={{
            background: "#fff",
            borderRadius: 24,
            boxShadow: "0 32px 80px rgba(0,0,0,0.2)",
            width: "100%",
            maxWidth: 420,
            border: "1px solid rgba(0,0,0,0.06)",
            overflow: "hidden",
          }}
        >
          {/* ── Header ── */}
          <div style={{
            padding: "22px 22px 18px",
            borderBottom: "1px solid rgba(0,0,0,0.05)",
            display: "flex", alignItems: "flex-start", justifyContent: "space-between",
          }}>
            <div>
              {/* Icon */}
              <div style={{
                width: 40, height: 40, borderRadius: 11,
                background: "#0c1a12",
                display: "flex", alignItems: "center", justifyContent: "center",
                marginBottom: 14,
              }}>
                <ShieldCheck size={18} color="#4ade80" />
              </div>

              {/* Label */}
              <p style={{
                fontSize: 10, fontWeight: 500, letterSpacing: "0.2em",
                textTransform: "uppercase", color: "#059669", marginBottom: 5,
              }}>
                KYC Verification
              </p>

              {/* Heading */}
              <h2 style={{
                fontFamily: "'Instrument Serif', serif",
                fontWeight: 400, fontSize: 24,
                letterSpacing: "-0.02em", color: "#0c1a12",
                margin: 0, marginBottom: 2, lineHeight: 1.2,
              }}>
                {policy.name}
              </h2>
              <p style={{ fontSize: 12, fontWeight: 300, color: "#94a3b8" }}>
                {householdName}
              </p>
            </div>

            {/* Close + counter */}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
              <button
                onClick={onClose}
                style={{
                  width: 30, height: 30, borderRadius: 99,
                  background: "#f8faf9", border: "1px solid rgba(0,0,0,0.07)",
                  cursor: "pointer", display: "flex",
                  alignItems: "center", justifyContent: "center",
                }}
              >
                <X size={14} color="#64748b" />
              </button>
              {/* Progress pill */}
              <div style={{
                display: "flex", alignItems: "center", gap: 5,
                background: validCount === docs.length
                  ? "rgba(5,150,105,0.09)"
                  : "rgba(0,0,0,0.04)",
                borderRadius: 999,
                padding: "4px 10px",
                border: `1px solid ${validCount === docs.length ? "rgba(5,150,105,0.2)" : "rgba(0,0,0,0.06)"}`,
              }}>
                {/* Mini dots */}
                {docs.map((d, i) => (
                  <div key={i} style={{
                    width: 6, height: 6, borderRadius: "50%",
                    background: d.status === "valid"
                      ? "#059669"
                      : d.status === "invalid"
                      ? "#ef4444"
                      : d.status === "uploading"
                      ? "#059669"
                      : "#cbd5e1",
                    transition: "background 0.2s",
                  }} />
                ))}
                <span style={{
                  fontSize: 10, fontWeight: 500,
                  color: validCount === docs.length ? "#059669" : "#94a3b8",
                  marginLeft: 2,
                }}>
                  {validCount}/{docs.length}
                </span>
              </div>
            </div>
          </div>

          {/* ── Doc cards ── */}
          <div style={{ padding: "18px 18px 0", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {docs.map((doc, idx) => {
              const colors = statusColor[doc.status];
              return (
                <div
                  key={doc.docType}
                  style={{
                    border: `1px solid ${colors.border}`,
                    borderRadius: 16,
                    overflow: "hidden",
                    display: "flex",
                    flexDirection: "column",
                    transition: "border-color 0.2s",
                  }}
                >
                  {/* Card header */}
                  <div style={{
                    background: colors.headerBg,
                    padding: "9px 11px",
                    display: "flex", alignItems: "center", gap: 7,
                    borderBottom: `1px solid ${colors.border}`,
                  }}>
                    <StatusIcon status={doc.status} />
                    <span style={{ fontSize: 11, fontWeight: 500, color: "#0c1a12", lineHeight: 1.3 }}>
                      {doc.docType}
                    </span>
                  </div>

                  {/* Card body */}
                  <div style={{
                    padding: 10, background: "#fff",
                    flex: 1, display: "flex", flexDirection: "column", gap: 8,
                  }}>
                    {doc.preview ? (
                      <>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={doc.preview}
                          alt="Document preview"
                          style={{
                            width: "100%", height: 80,
                            objectFit: "cover",
                            borderRadius: 10,
                            border: "1px solid rgba(0,0,0,0.06)",
                            display: "block",
                          }}
                        />

                        {doc.status === "valid" && (
                          <p style={{
                            fontSize: 11, fontWeight: 400,
                            color: "#059669", lineHeight: 1.4,
                          }}>
                            ✓ {doc.extractedName}
                          </p>
                        )}

                        {doc.status === "invalid" && (
                          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                            <p style={{ fontSize: 11, fontWeight: 300, color: "#ef4444", lineHeight: 1.4 }}>
                              {doc.reason}
                            </p>
                            <button
                              className="retry-btn"
                              onClick={() => fileInputRefs.current[idx]?.click()}
                              style={{
                                background: "none", border: "none",
                                fontSize: 11, fontWeight: 500,
                                color: "#ef4444", cursor: "pointer",
                                textAlign: "left", padding: 0,
                                fontFamily: "'DM Sans', sans-serif",
                              }}
                            >
                              Try again →
                            </button>
                          </div>
                        )}

                        {doc.status === "uploading" && (
                          <p style={{ fontSize: 11, fontWeight: 300, color: "#94a3b8" }}>
                            Validating…
                          </p>
                        )}
                      </>
                    ) : (
                      <button
                        className="upload-zone"
                        onClick={() => fileInputRefs.current[idx]?.click()}
                        style={{
                          width: "100%", height: 80,
                          border: "1.5px dashed rgba(0,0,0,0.12)",
                          borderRadius: 10,
                          background: "none",
                          cursor: "pointer",
                          display: "flex", flexDirection: "column",
                          alignItems: "center", justifyContent: "center", gap: 5,
                          transition: "border-color 0.2s, background 0.2s",
                        }}
                      >
                        <Upload size={15} color="#94a3b8" className="upload-icon" style={{ transition: "color 0.2s" }} />
                        <span style={{
                          fontSize: 11, fontWeight: 400, color: "#94a3b8",
                          fontFamily: "'DM Sans', sans-serif",
                          transition: "color 0.2s",
                        }}>
                          Upload
                        </span>
                      </button>
                    )}
                  </div>

                  <input
                    ref={(el) => { fileInputRefs.current[idx] = el; }}
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    style={{ display: "none" }}
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleFileSelect(idx, file);
                      e.target.value = "";
                    }}
                  />
                </div>
              );
            })}
          </div>

          {/* ── Footer ── */}
          <div style={{ padding: "16px 18px 20px" }}>
            {allValid ? (
              <button
                onClick={handleIssue}
                disabled={issuing}
                className="issue-btn"
                style={{
                  width: "100%",
                  background: "#059669",
                  color: "#fff",
                  border: "none",
                  borderRadius: 999,
                  padding: "13px 24px",
                  fontSize: 14, fontWeight: 500,
                  fontFamily: "'DM Sans', sans-serif",
                  cursor: issuing ? "default" : "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                  transition: "background 0.15s, transform 0.15s",
                }}
              >
                {issuing
                  ? <><Loader2 size={15} className="animate-spin" /> Issuing Policy…</>
                  : <><ShieldCheck size={15} /> Issue Policy</>
                }
              </button>
            ) : (
              <div style={{ textAlign: "center" }}>
                <p style={{ fontSize: 11, fontWeight: 300, color: "#94a3b8" }}>
                  {anyUploading
                    ? "Validating documents…"
                    : "Upload both documents to proceed"
                  }
                </p>
                {/* Progress bar */}
                <div style={{
                  height: 3, background: "rgba(0,0,0,0.05)",
                  borderRadius: 99, marginTop: 10, overflow: "hidden",
                }}>
                  <div style={{
                    height: "100%",
                    width: `${(validCount / docs.length) * 100}%`,
                    background: "#059669",
                    borderRadius: 99,
                    transition: "width 0.4s ease",
                  }} />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}