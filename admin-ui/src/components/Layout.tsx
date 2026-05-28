import { NavLink, Outlet } from "react-router-dom";
import {
  Activity,
  BellRing,
  Boxes,
  Database,
  FlaskConical,
  LayoutDashboard,
  LogOut,
  MessageSquare,
  Rocket,
  Sparkles,
  Users,
} from "lucide-react";
import { useAuth } from "@/auth/AuthContext";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/conversations", label: "Q&A History", icon: MessageSquare },
  { to: "/drift", label: "Drift", icon: BellRing },
  { to: "/experiments", label: "Experiments", icon: FlaskConical },
  { to: "/jobs/new", label: "Дообучение", icon: Rocket },
  { to: "/jobs", label: "Jobs", icon: Activity },
  { to: "/models", label: "Models", icon: Boxes },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/users", label: "Users", icon: Users },
];

export function Layout() {
  const { logout } = useAuth();

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[260px_1fr]">
      <aside className="border-b border-[var(--border)] bg-[#0d1219]/90 backdrop-blur lg:min-h-screen lg:border-b-0 lg:border-r">
        <div className="px-5 py-6">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[var(--accent-soft)] text-[var(--accent)]">
              <Sparkles size={20} />
            </div>
            <div>
              <div className="text-lg font-semibold">Admin Studio</div>
              <div className="text-sm text-[var(--muted)]">MLOps control plane</div>
            </div>
          </div>
        </div>
        <nav className="px-3 pb-3">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                [
                  "mb-1 flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition",
                  isActive
                    ? "bg-[var(--accent-soft)] text-[#dbeafe]"
                    : "text-[var(--muted)] hover:bg-white/5 hover:text-white",
                ].join(" ")
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="px-3 pb-6">
          <button type="button" className="btn btn-secondary w-full" onClick={logout}>
            <LogOut size={16} />
            Выйти
          </button>
        </div>
      </aside>

      <main className="px-4 py-6 sm:px-6 lg:px-8">
        <Outlet />
      </main>
    </div>
  );
}
