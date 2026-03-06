import Link from "next/link";
import { SignedIn, SignedOut, UserButton } from "@clerk/nextjs";
import { Button } from "../ui/button";

// Navbar is shown on all marketing pages
// SignedIn/SignedOut are Clerk components that conditionally render based on auth state
export default function Navbar() {
  return (
    <nav className="fixed top-0 w-full border-b border-slate-100 bg-white/80 backdrop-blur-sm z-50">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        
        {/* Logo */}
        <Link href="/" className="font-semibold text-xl tracking-tight">
          Sure<span className="text-emerald-600">Im</span>
        </Link>

        {/* Nav links — hidden on mobile */}
        <div className="hidden md:flex items-center gap-8 text-sm text-slate-600">
          <Link href="#problem" className="hover:text-slate-900 transition-colors">Problem</Link>
          <Link href="#solution" className="hover:text-slate-900 transition-colors">Solution</Link>
          <Link href="#features" className="hover:text-slate-900 transition-colors">Features</Link>
        </div>

        {/* Auth buttons */}
        <div className="flex items-center gap-3">
          <SignedOut>
            <Link href="/sign-in">
              <Button variant="ghost" size="sm">Sign in</Button>
            </Link>
            <Link href="/sign-up">
              <Button size="sm" className="bg-emerald-600 hover:bg-emerald-700 text-white">
                Get Started
              </Button>
            </Link>
          </SignedOut>

          {/* Shows user avatar + dropdown when signed in */}
          <SignedIn>
            <Link href="/dashboard">
              <Button variant="ghost" size="sm">Dashboard</Button>
            </Link>
            <UserButton afterSignOutUrl="/" />
          </SignedIn>
        </div>

      </div>
    </nav>
  );
}