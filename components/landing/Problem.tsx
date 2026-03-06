import { ShieldOff, MapPin, Wifi } from "lucide-react";

const problems = [
  {
    icon: ShieldOff,
    title: "No Awareness",
    description:
      "Insurance benefits aren't immediate, making the concept feel irrelevant to rural households with no sustained awareness campaigns reaching them.",
  },
  {
    icon: MapPin,
    title: "No Access",
    description:
      "Insurance companies are concentrated in urban centers. Rural communities simply don't have reliable access to insurance products or representatives.",
  },
  {
    icon: Wifi,
    title: "No Trust",
    description:
      "Despite rising internet penetration, people prefer offline interactions. Fear of claim rejections and low financial literacy fuel deep skepticism.",
  },
];

export default function Problem() {
  return (
    <section id="problem" className="py-24 bg-slate-50">
      <div className="max-w-6xl mx-auto px-6">
        <div className="text-center mb-16">
          <p className="text-emerald-600 font-medium text-sm uppercase tracking-widest mb-3">
            The Problem
          </p>
          <h2 className="text-4xl font-bold text-slate-900 tracking-tight">
            61% of rural India is uninsured.
            <br />
            <span className="text-slate-400">Here's why.</span>
          </h2>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {problems.map((problem) => (
            <div
              key={problem.title}
              className="bg-white rounded-2xl p-8 border border-slate-100 hover:border-emerald-100 hover:shadow-sm transition-all"
            >
              <div className="w-10 h-10 bg-red-50 rounded-xl flex items-center justify-center mb-5">
                <problem.icon className="w-5 h-5 text-red-500" />
              </div>
              <h3 className="font-semibold text-slate-900 text-lg mb-3">
                {problem.title}
              </h3>
              <p className="text-slate-500 leading-relaxed text-sm">
                {problem.description}
              </p>
            </div>
          ))}
        </div>

        <div className="mt-12 text-center">
          <p className="text-slate-400 text-sm">
            Source: Jio Insurance Brokers — Rural Insurance Report
          </p>
        </div>
      </div>
    </section>
  );
}