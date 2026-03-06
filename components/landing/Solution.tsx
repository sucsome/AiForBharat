const steps = [
  {
    step: "01",
    title: "Agent gets onboarded",
    description:
      "A local farmer, kirana owner, or domestic worker signs up as an agent. No prior insurance knowledge needed — our AI fills that gap instantly.",
  },
  {
    step: "02",
    title: "AI recommends the right policy",
    description:
      "Agent inputs household details. Our AI analyzes income, family size, and risk exposure to recommend the most suitable policies in the local language.",
  },
  {
    step: "03",
    title: "Policy gets issued",
    description:
      "Agent guides the family through the process. Documents are collected, verified, and transmitted to the insurer — all from a mobile device.",
  },
];

export default function Solution() {
  return (
    <section id="solution" className="py-24 bg-white">
      <div className="max-w-6xl mx-auto px-6">
        <div className="text-center mb-16">
          <p className="text-emerald-600 font-medium text-sm uppercase tracking-widest mb-3">
            The Solution
          </p>
          <h2 className="text-4xl font-bold text-slate-900 tracking-tight">
            Local agents. AI-powered.
            <br />
            <span className="text-slate-400">Trust at scale.</span>
          </h2>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {steps.map((step) => (
            <div key={step.step} className="relative">
              <p className="text-6xl font-bold text-slate-100 mb-4">{step.step}</p>
              <h3 className="font-semibold text-slate-900 text-lg mb-3">{step.title}</h3>
              <p className="text-slate-500 text-sm leading-relaxed">{step.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}