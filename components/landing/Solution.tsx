"use client";

import { useEffect, useRef } from "react";
import { UserPlus, Sparkles, FileCheck } from "lucide-react";

const steps = [
  {
    step: "01",
    icon: UserPlus,
    title: "Agent gets onboarded",
    description:
      "A local farmer, kirana owner, or domestic worker signs up as an agent. No prior insurance knowledge needed — our AI fills that gap instantly.",
    bg: "#fff",
    dark: false,
    large: true, // spans top full width
  },
  {
    step: "02",
    icon: Sparkles,
    title: "AI recommends the right policy",
    description:
      "Agent inputs household details. Our AI analyzes income, family size, and risk exposure to recommend the most suitable policies in the local language.",
    bg: "#0c1a12",
    dark: true,
    large: false,
  },
  {
    step: "03",
    icon: FileCheck,
    title: "Policy gets issued",
    description:
      "Agent guides the family through the process. Documents are collected, verified, and transmitted to the insurer — all from a mobile device.",
    bg: "#fff",
    dark: false,
    large: false,
  },
];

export default function Solution() {
  const headingRef = useRef<HTMLDivElement>(null);
  const card0Ref   = useRef<HTMLDivElement>(null);
  const card1Ref   = useRef<HTMLDivElement>(null);
  const card2Ref   = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const init = async () => {
      const gsap = (await import("gsap")).default;
      const { ScrollTrigger } = await import("gsap/ScrollTrigger");
      gsap.registerPlugin(ScrollTrigger);

      // heading
      gsap.fromTo(headingRef.current,
        { y: 28, opacity: 0 },
        { y: 0, opacity: 1, duration: 1.0, ease: "expo.out",
          scrollTrigger: { trigger: headingRef.current, start: "top 80%" } }
      );

      // clip reveal for each card
      [
        { ref: card0Ref, delay: 0 },
        { ref: card1Ref, delay: 0.1 },
        { ref: card2Ref, delay: 0.2 },
      ].forEach(({ ref, delay }) => {
        const el = ref.current;
        if (!el) return;
        const inner = el.querySelector<HTMLElement>(".card-inner");
        if (!inner) return;
        gsap.fromTo(inner,
          { y: "105%", skewY: 1 },
          {
            y: "0%", skewY: 0,
            duration: 1.15,
            ease: "expo.out",
            delay,
            scrollTrigger: { trigger: el, start: "top 85%" },
          }
        );
      });
    };

    init();
  }, []);

  const clipWrap = (extra: React.CSSProperties = {}): React.CSSProperties => ({
    overflow: "hidden",
    borderRadius: 20,
    border: "1px solid rgba(0,0,0,0.06)",
    ...extra,
  });

  return (
    <section id="solution" style={{ padding: "96px 0", backgroundColor: "#fff" }}>
      <div style={{ maxWidth: 960, margin: "0 auto", padding: "0 24px" }}>

        {/* Heading */}
        <div ref={headingRef} style={{ textAlign: "center", marginBottom: 48, opacity: 0 }}>
          <p style={{
            fontFamily: "'DM Sans', sans-serif", fontWeight: 500, fontSize: 11,
            letterSpacing: "0.2em", textTransform: "uppercase", color: "#059669", marginBottom: 14,
          }}>The Solution</p>
          <h2 style={{
            fontFamily: "'Instrument Serif', serif", fontWeight: 400,
            fontSize: "clamp(32px, 5vw, 52px)", letterSpacing: "-0.02em",
            lineHeight: 1.1, color: "#0c1a12", margin: 0,
          }}>
            Local agents. AI-powered.
            <br />
            <span style={{ color: "#94a3b8" }}>Trust at scale.</span>
          </h2>
        </div>

        {/* Bento: top wide card, bottom two side by side */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

          {/* Row 1 — wide card (step 01) */}
          <div ref={card0Ref} style={clipWrap()}>
            <div className="card-inner" style={{
              backgroundColor: "#f8faf9",
              padding: "48px 44px",
              display: "flex",
              alignItems: "center",
              gap: 48,
              willChange: "transform",
            }}>
              {/* Big step number */}
              <span style={{
                fontFamily: "'Instrument Serif', serif",
                fontWeight: 400,
                fontSize: 120,
                letterSpacing: "-0.05em",
                color: "rgba(5,150,105,0.12)",
                lineHeight: 1,
                flexShrink: 0,
                userSelect: "none",
              }}>01</span>

              <div>
                <div style={{
                  width: 44, height: 44, borderRadius: 12, backgroundColor: "#d1fae5",
                  display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18,
                }}>
                  <UserPlus size={20} color="#059669" />
                </div>
                <h3 style={{
                  fontFamily: "'Instrument Serif', serif", fontWeight: 400, fontSize: 26,
                  letterSpacing: "-0.01em", color: "#0c1a12", marginBottom: 12,
                }}>Agent gets onboarded</h3>
                <p style={{
                  fontFamily: "'DM Sans', sans-serif", fontWeight: 300, fontSize: 15,
                  color: "#64748b", lineHeight: 1.75, margin: 0, maxWidth: 520,
                }}>
                  A local farmer, kirana owner, or domestic worker signs up as an agent.
                  No prior insurance knowledge needed — our AI fills that gap instantly.
                </p>
              </div>
            </div>
          </div>

          {/* Row 2 — two cards side by side */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>

            {/* Step 02 — dark */}
            <div ref={card1Ref} style={clipWrap()}>
              <div className="card-inner" style={{
                backgroundColor: "#0c1a12",
                padding: "36px 32px",
                willChange: "transform",
                minHeight: 280,
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
              }}>
                <div>
                  <div style={{
                    width: 44, height: 44, borderRadius: 12,
                    backgroundColor: "rgba(255,255,255,0.08)",
                    display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 20,
                  }}>
                    <Sparkles size={20} color="#4ade80" />
                  </div>
                  <h3 style={{
                    fontFamily: "'Instrument Serif', serif", fontWeight: 400, fontSize: 22,
                    letterSpacing: "-0.01em", color: "#fff", marginBottom: 12,
                  }}>AI recommends the right policy</h3>
                  <p style={{
                    fontFamily: "'DM Sans', sans-serif", fontWeight: 300, fontSize: 14,
                    color: "rgba(255,255,255,0.5)", lineHeight: 1.75, margin: 0,
                  }}>
                    Agent inputs household details. Our AI analyzes income, family size, and
                    risk exposure to recommend the most suitable policies in the local language.
                  </p>
                </div>
                <span style={{
                  fontFamily: "'Instrument Serif', serif", fontWeight: 400,
                  fontSize: 72, letterSpacing: "-0.05em",
                  color: "rgba(74,222,128,0.15)", lineHeight: 1,
                  display: "block", marginTop: 16, userSelect: "none",
                }}>02</span>
              </div>
            </div>

            {/* Step 03 — light */}
            <div ref={card2Ref} style={clipWrap()}>
              <div className="card-inner" style={{
                backgroundColor: "#fff",
                padding: "36px 32px",
                willChange: "transform",
                minHeight: 280,
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
              }}>
                <div>
                  <div style={{
                    width: 44, height: 44, borderRadius: 12, backgroundColor: "#d1fae5",
                    display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 20,
                  }}>
                    <FileCheck size={20} color="#059669" />
                  </div>
                  <h3 style={{
                    fontFamily: "'Instrument Serif', serif", fontWeight: 400, fontSize: 22,
                    letterSpacing: "-0.01em", color: "#0c1a12", marginBottom: 12,
                  }}>Policy gets issued</h3>
                  <p style={{
                    fontFamily: "'DM Sans', sans-serif", fontWeight: 300, fontSize: 14,
                    color: "#64748b", lineHeight: 1.75, margin: 0,
                  }}>
                    Agent guides the family through the process. Documents are collected,
                    verified, and transmitted to the insurer — all from a mobile device.
                  </p>
                </div>
                <span style={{
                  fontFamily: "'Instrument Serif', serif", fontWeight: 400,
                  fontSize: 72, letterSpacing: "-0.05em",
                  color: "rgba(5,150,105,0.1)", lineHeight: 1,
                  display: "block", marginTop: 16, userSelect: "none",
                }}>03</span>
              </div>
            </div>

          </div>
        </div>

      </div>
    </section>
  );
}