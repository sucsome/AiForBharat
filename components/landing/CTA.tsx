"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";
import { ArrowRight } from "lucide-react";

export default function CTA() {
  const headingRef = useRef<HTMLDivElement>(null);
  const subRef     = useRef<HTMLParagraphElement>(null);
  const btnRef     = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const init = async () => {
      const gsap = (await import("gsap")).default;
      const { ScrollTrigger } = await import("gsap/ScrollTrigger");
      gsap.registerPlugin(ScrollTrigger);

      const inner = headingRef.current?.querySelector<HTMLElement>(".clip-inner");
      if (inner) {
        gsap.fromTo(inner,
          { y: "105%", skewY: 1 },
          { y: "0%", skewY: 0, duration: 1.1, ease: "expo.out",
            scrollTrigger: { trigger: headingRef.current, start: "top 82%" } }
        );
      }

      gsap.fromTo(subRef.current,
        { y: 16, opacity: 0 },
        { y: 0, opacity: 1, duration: 1.0, ease: "expo.out", delay: 0.15,
          scrollTrigger: { trigger: subRef.current, start: "top 85%" } }
      );

      gsap.fromTo(btnRef.current,
        { y: 14, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.9, ease: "expo.out", delay: 0.28,
          scrollTrigger: { trigger: btnRef.current, start: "top 88%" } }
      );
    };
    init();
  }, []);

  return (
    <>
      {/* ── CTA ── */}
      <section style={{
        position: "relative",
        padding: "112px 24px",
        backgroundColor: "#0c1a12",
        overflow: "hidden",
        textAlign: "center",
      }}>

        {/* grain overlay */}
        <div style={{
          position: "absolute", inset: 0, pointerEvents: "none", zIndex: 0,
          opacity: 0.045,
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
          backgroundSize: "256px 256px",
          mixBlendMode: "overlay",
        }} />

        {/* soft radial glow */}
        <div style={{
          position: "absolute", inset: 0, pointerEvents: "none", zIndex: 0,
          background: "radial-gradient(ellipse 70% 60% at 50% 50%, rgba(5,150,105,0.18) 0%, transparent 70%)",
        }} />

        <div style={{ position: "relative", zIndex: 1, maxWidth: 640, margin: "0 auto" }}>

          {/* label */}
          <p style={{
            fontFamily: "'DM Sans', sans-serif", fontWeight: 500, fontSize: 11,
            letterSpacing: "0.2em", textTransform: "uppercase",
            color: "rgba(74,222,128,0.7)", marginBottom: 20,
          }}>Join the movement</p>

          {/* headline — clip reveal */}
          <div ref={headingRef} style={{ overflow: "hidden", marginBottom: 24 }}>
            <div className="clip-inner" style={{ willChange: "transform" }}>
              <h2 style={{
                fontFamily: "'Instrument Serif', serif", fontWeight: 400,
                fontSize: "clamp(36px, 6vw, 64px)", letterSpacing: "-0.025em",
                lineHeight: 1.08, color: "#fff", margin: 0,
              }}>
                Ready to make
                <br />
                <span style={{ color: "#4ade80", fontStyle: "italic" }}>a difference?</span>
              </h2>
            </div>
          </div>

          {/* subtext */}
          <p ref={subRef} style={{
            fontFamily: "'DM Sans', sans-serif", fontWeight: 300,
            fontSize: "clamp(15px, 2vw, 18px)", color: "rgba(255,255,255,0.45)",
            lineHeight: 1.75, margin: "0 auto 40px", maxWidth: 460, opacity: 0,
          }}>
            Join our network of agents bringing financial protection to rural India.
            No experience needed — just your community connection.
          </p>

          {/* CTA button */}
          <div ref={btnRef} style={{ opacity: 0 }}>
            <Link href="/sign-up">
              <button
                style={{
                  display: "inline-flex", alignItems: "center", gap: 8,
                  backgroundColor: "#4ade80", color: "#0c1a12",
                  fontFamily: "'DM Sans', sans-serif", fontWeight: 600,
                  fontSize: 16, padding: "14px 30px", borderRadius: 999,
                  border: "none", cursor: "pointer",
                  transition: "transform 0.15s ease, background 0.15s ease",
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.backgroundColor = "#86efac";
                  e.currentTarget.style.transform = "translateY(-2px)";
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.backgroundColor = "#4ade80";
                  e.currentTarget.style.transform = "translateY(0)";
                }}
              >
                Join as an Agent <ArrowRight size={16} />
              </button>
            </Link>
          </div>

        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer style={{
        backgroundColor: "#080f0a",
        borderTop: "1px solid rgba(255,255,255,0.05)",
        padding: "32px 24px",
      }}>
        <div style={{
          maxWidth: 960, margin: "0 auto",
          display: "flex", flexDirection: "row",
          alignItems: "center", justifyContent: "space-between",
          flexWrap: "wrap", gap: 12,
        }}>
          {/* logo */}
          <p style={{
            fontFamily: "'Instrument Serif', serif", fontWeight: 400,
            fontSize: 20, letterSpacing: "-0.02em",
            color: "#fff", margin: 0,
          }}>
            Sure<span style={{ color: "#4ade80" }}>LM</span>
          </p>

          {/* links */}
          <div style={{ display: "flex", gap: 24 }}>
            {["Problem", "Solution", "Features"].map(l => (
              <Link key={l} href={`#${l.toLowerCase()}`} style={{
                fontFamily: "'DM Sans', sans-serif", fontWeight: 300,
                fontSize: 13, color: "rgba(255,255,255,0.3)",
                textDecoration: "none", transition: "color 0.15s",
              }}
              onMouseEnter={e => (e.currentTarget.style.color = "rgba(255,255,255,0.7)")}
              onMouseLeave={e => (e.currentTarget.style.color = "rgba(255,255,255,0.3)")}
              >{l}</Link>
            ))}
          </div>

          {/* copyright */}
          <p style={{
            fontFamily: "'DM Sans', sans-serif", fontWeight: 300,
            fontSize: 12, color: "rgba(255,255,255,0.2)", margin: 0,
          }}>
            © {new Date().getFullYear()} SureLM. Built for rural India.
          </p>
        </div>
      </footer>
    </>
  );
}