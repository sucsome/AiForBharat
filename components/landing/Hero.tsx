import Link from "next/link";
import { Button } from '../ui/button';
import { ArrowRight, ShieldCheck } from "lucide-react";

export default function Hero() {
  return (
    <section className="min-h-screen flex items-center justify-center bg-white pt-16">
      <div className="max-w-6xl mx-auto px-6 py-24 text-center">

        {/* Small badge above headline */}
        <div className="inline-flex items-center gap-2 bg-emerald-50 text-emerald-700 text-sm font-medium px-4 py-1.5 rounded-full mb-8">
          <ShieldCheck className="w-4 h-4" />
          AI-powered insurance for rural India
        </div>

        {/* Main headline */}
        <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-slate-900 mb-6 leading-tight">
          Financial protection
          <br />
          <span className="text-emerald-600">for every household</span>
        </h1>

        {/* Subtext */}
        <p className="text-lg md:text-xl text-slate-500 max-w-2xl mx-auto mb-10 leading-relaxed">
          We empower local agents with AI to bring the right insurance policies 
          to rural families — in their language, at their doorstep.
        </p>

        {/* CTAs */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link href="/sign-up">
            <Button 
              size="lg" 
              className="bg-emerald-600 hover:bg-emerald-700 text-white px-8 h-12 text-base"
            >
              Join as an Agent
              <ArrowRight className="ml-2 w-4 h-4" />
            </Button>
          </Link>
          <Link href="#problem">
            <Button 
              size="lg" 
              variant="outline" 
              className="px-8 h-12 text-base border-slate-200 text-slate-600 hover:bg-slate-50"
            >
              Learn more
            </Button>
          </Link>
        </div>

        {/* Social proof stats */}
        <div className="mt-20 grid grid-cols-3 gap-8 max-w-lg mx-auto">
          <div>
            <p className="text-3xl font-bold text-slate-900">65%</p>
            <p className="text-sm text-slate-500 mt-1">of India lives in rural areas</p>
          </div>
          <div className="border-x border-slate-100">
            <p className="text-3xl font-bold text-slate-900">61%</p>
            <p className="text-sm text-slate-500 mt-1">still lack health insurance</p>
          </div>
          <div>
            <p className="text-3xl font-bold text-slate-900">60Cr</p>
            <p className="text-sm text-slate-500 mt-1">people we can reach</p>
          </div>
        </div>

      </div>
    </section>
  );
}