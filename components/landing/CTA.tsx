import Link from "next/link";

export default function CTA() {
  return (
    <>
      {/* CTA Section */}
      <section className="py-24 bg-emerald-600">
        <div className="max-w-6xl mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold text-white tracking-tight mb-4">
            Ready to make a difference?
          </h2>
          <p className="text-emerald-100 text-lg mb-8 max-w-xl mx-auto">
            Join our network of agents bringing financial protection to rural India.
            No experience needed — just your community connection.
          </p>
          <Link
            href="/sign-up"
            className="inline-block bg-white text-emerald-600 font-semibold px-8 py-3.5 rounded-xl hover:bg-emerald-50 transition-colors"
          >
            Join as an Agent →
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-100 py-8">
        <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="font-semibold text-slate-900">
            Sure<span className="text-emerald-600">Im</span>
          </p>
          <p className="text-slate-400 text-sm">
            © {new Date().getFullYear()} SureIm. Built for rural India.
          </p>
        </div>
      </footer>
    </>
  );
}