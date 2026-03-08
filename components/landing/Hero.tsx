"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";
import gsap from "gsap";
import { ArrowRight } from "lucide-react";

const SPLASH_DURATION = 8.0;

function Reveal({
  children,
  delay = 0,
}: {
  children: React.ReactNode;
  delay?: number;
}) {
  const innerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!innerRef.current) return;
    gsap.fromTo(
      innerRef.current,
      { y: "105%", skewY: 1 },
      {
        y: "0%",
        skewY: 0,
        duration: 1.1,
        ease: "expo.out",
        delay: SPLASH_DURATION + delay,
      }
    );
  }, [delay]);

  return (
    <span style={{ display: "block", overflow: "hidden", lineHeight: 1.1 }}>
      <span ref={innerRef} style={{ display: "block", willChange: "transform" }}>
        {children}
      </span>
    </span>
  );
}

export default function Hero() {
  const lineRef   = useRef<HTMLDivElement>(null);
  const subRef    = useRef<HTMLParagraphElement>(null);
  const ctaRef    = useRef<HTMLDivElement>(null);
  const statItems = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const S = SPLASH_DURATION;

    // decorative line draws in after headline
    gsap.fromTo(lineRef.current,
      { scaleX: 0, opacity: 0 },
      { scaleX: 1, opacity: 1, duration: 1.0, ease: "expo.out", delay: S + 0.38, transformOrigin: "left center" }
    );

    gsap.fromTo(subRef.current,
      { y: 14, opacity: 0 },
      { y: 0, opacity: 1, duration: 1.0, ease: "expo.out", delay: S + 0.52 }
    );

    gsap.fromTo(ctaRef.current,
      { y: 14, opacity: 0 },
      { y: 0, opacity: 1, duration: 1.0, ease: "expo.out", delay: S + 0.68 }
    );

    gsap.fromTo(
      statItems.current.filter(Boolean),
      { y: 16, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.9, ease: "expo.out", delay: S + 0.88, stagger: 0.1 }
    );
  }, []);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

        .hero-grain {
          position: relative;
          background-color: #fafaf9;
        }
        .hero-grain::before {
          content: "";
          position: fixed;
          inset: 0;
          pointer-events: none;
          z-index: 0;
          opacity: 0.028;
          background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
          background-size: 256px 256px;
          mix-blend-mode: multiply;
        }
        .hero-grain > * { position: relative; z-index: 1; }
      `}</style>

      <section className="hero-grain min-h-screen flex flex-col items-center justify-center pt-40">
        <div className="w-full max-w-3xl mx-auto px-6 text-center">

          {/* Headline */}
          <h1
            style={{
              fontFamily:    "'Instrument Serif', serif",
              fontWeight:    400,
              fontSize:      "clamp(50px, 8.5vw, 90px)",
              lineHeight:    1.08,
              letterSpacing: "-0.02em",
              color:         "#0c1a12",
              marginBottom:  0,
            }}
          >
            <Reveal delay={0.0}>Financial protection</Reveal>
            <Reveal delay={0.16}>
              <span style={{ color: "#059669", fontStyle: "italic" }}>
                for every household
              </span>
            </Reveal>
          </h1>

          {/* Decorative line */}
          <div
            ref={lineRef}
            style={{
              height:          1,
              background:      "linear-gradient(90deg, #059669 0%, rgba(5,150,105,0.15) 60%, transparent 100%)",
              margin:          "20px auto 28px",
              maxWidth:        480,
              opacity:         0,
              transformOrigin: "left center",
            }}
          />

          {/* Subtext */}
          <p
            ref={subRef}
            style={{
              fontFamily: "'DM Sans', sans-serif",
              fontWeight: 400,
              fontSize:   "clamp(15px, 1.8vw, 19px)",
              color:      "#64748b",
              maxWidth:   500,
              margin:     "0 auto 40px",
              lineHeight: 1.7,
              opacity:    0,
            }}
          >
            We empower local agents with AI to bring the right insurance
            policies to rural families — in their language, at their doorstep.
          </p>

          {/* CTAs */}
          <div
            ref={ctaRef}
            style={{
              display:        "flex",
              alignItems:     "center",
              justifyContent: "center",
              gap:            10,
              flexWrap:       "wrap",
              opacity:        0,
            }}
          >
            <Link href="/sign-up">
              <button
                style={{
                  display:         "inline-flex",
                  alignItems:      "center",
                  gap:             8,
                  backgroundColor: "#059669",
                  color:           "#fff",
                  fontFamily:      "'DM Sans', sans-serif",
                  fontWeight:      500,
                  fontSize:        16,
                  padding:         "13px 26px",
                  borderRadius:    999,
                  border:          "none",
                  cursor:          "pointer",
                  transition:      "background 0.2s ease, transform 0.15s ease",
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.backgroundColor = "#047857";
                  e.currentTarget.style.transform = "translateY(-1px)";
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.backgroundColor = "#059669";
                  e.currentTarget.style.transform = "translateY(0)";
                }}
              >
                Join as an Agent <ArrowRight size={15} />
              </button>
            </Link>

            <Link href="#problem">
              <button
                style={{
                  display:         "inline-flex",
                  alignItems:      "center",
                  backgroundColor: "#0c1a12",
                  color:           "#fff",
                  fontFamily:      "'DM Sans', sans-serif",
                  fontWeight:      500,
                  fontSize:        16,
                  padding:         "13px 26px",
                  borderRadius:    999,
                  border:          "none",
                  cursor:          "pointer",
                  transition:      "background 0.2s ease, transform 0.15s ease",
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.backgroundColor = "#1c3326";
                  e.currentTarget.style.transform = "translateY(-1px)";
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.backgroundColor = "#0c1a12";
                  e.currentTarget.style.transform = "translateY(0)";
                }}
              >
                Learn more
              </button>
            </Link>
          </div>

          {/* Stats */}
          <div
            style={{
              display:             "grid",
              gridTemplateColumns: "1fr 1fr 1fr",
              gap:                 24,
              maxWidth:            400,
              margin:              "64px auto 0",
            }}
          >
            {[
              { value: "65%",  label: "of India lives in rural areas" },
              { value: "61%",  label: "still lack health insurance", border: true },
              { value: "60Cr", label: "people we can reach" },
            ].map(({ value, label, border }, i) => (
              <div
                key={value}
                ref={el => { statItems.current[i] = el; }}
                style={{
                  opacity:     0,
                  borderLeft:  border ? "1px solid #e2e8f0" : "none",
                  borderRight: border ? "1px solid #e2e8f0" : "none",
                  padding:     border ? "0 12px" : undefined,
                }}
              >
                <p style={{
                  fontFamily: "'Instrument Serif', serif",
                  fontWeight: 400,
                  fontSize:   30,
                  color:      "#0c1a12",
                  margin:     0,
                  letterSpacing: "-0.02em",
                }}>{value}</p>
                <p style={{
                  fontFamily: "'DM Sans', sans-serif",
                  fontWeight: 300,
                  fontSize:   11,
                  color:      "#94a3b8",
                  marginTop:  5,
                  lineHeight: 1.5,
                }}>{label}</p>
              </div>
            ))}
          </div>

        </div>
      </section>
    </>
  );
}