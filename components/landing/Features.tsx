import { Brain, Globe, UserCheck, FileSearch, MessageSquare, BarChart3 } from "lucide-react";

const features = [
  {
    icon: Brain,
    title: "AI Policy Guru",
    description: "Real-time policy recommendations powered by LLMs and RAG — agents get expert-level guidance instantly.",
  },
  {
    icon: Globe,
    title: "Multilingual",
    description: "Every interaction available in the user's mother tongue. Hindi, Tamil, Telugu, Marathi and more.",
  },
  {
    icon: UserCheck,
    title: "Local Agent Network",
    description: "Empower existing community members — farmers, shop owners — to become trusted insurance educators.",
  },
  {
    icon: FileSearch,
    title: "Smart Document Processing",
    description: "OCR-based document extraction and validation. No manual data entry, no errors, no rejections.",
  },
  {
    icon: MessageSquare,
    title: "Claims Assistance",
    description: "Step-by-step claims guidance in local language. Agents can support families through every stage.",
  },
  {
    icon: BarChart3,
    title: "Agent Dashboard",
    description: "Track leads, policies issued, and household status. Everything an agent needs in one place.",
  },
];

export default function Features() {
  return (
    <section id="features" className="py-24 bg-slate-50">
      <div className="max-w-6xl mx-auto px-6">
        <div className="text-center mb-16">
          <p className="text-emerald-600 font-medium text-sm uppercase tracking-widest mb-3">
            Features
          </p>
          <h2 className="text-4xl font-bold text-slate-900 tracking-tight">
            Everything an agent needs
            <br />
            <span className="text-slate-400">to protect a community.</span>
          </h2>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="bg-white rounded-2xl p-7 border border-slate-100 hover:border-emerald-100 hover:shadow-sm transition-all"
            >
              <div className="w-10 h-10 bg-emerald-50 rounded-xl flex items-center justify-center mb-5">
                <feature.icon className="w-5 h-5 text-emerald-600" />
              </div>
              <h3 className="font-semibold text-slate-900 mb-2">{feature.title}</h3>
              <p className="text-slate-500 text-sm leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}