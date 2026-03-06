import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatINR(amount: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(amount);
}

export function formatIndianNumber(num: number) {
  if (num >= 10000000) return `${(num / 10000000).toFixed(1)} Crore`;
  if (num >= 100000) return `${(num / 100000).toFixed(1)} Lakh`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
}