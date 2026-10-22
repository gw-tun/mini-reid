import type { Metadata } from "next";

import { DashboardOverview } from "@/components/dashboard-overview";

export const metadata: Metadata = {
  title: "Dashboard",
};

export default function Home() {
  return <DashboardOverview />;
}
