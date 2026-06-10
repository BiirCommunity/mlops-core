import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider } from "@/auth/AuthContext";
import { ProtectedRoute } from "@/auth/ProtectedRoute";
import { Layout } from "@/components/Layout";
import { ConversationsPage } from "@/pages/ConversationsPage";
import { DashboardPage } from "@/pages/DashboardPage";
import { DatasetsPage } from "@/pages/DatasetsPage";
import { DriftAlertsPage } from "@/pages/DriftAlertsPage";
import { ExperimentsPage } from "@/pages/ExperimentsPage";
import { JobDetailPage } from "@/pages/JobDetailPage";
import { JobsPage } from "@/pages/JobsPage";
import { LoginPage } from "@/pages/LoginPage";
import { ModelsPage } from "@/pages/ModelsPage";
import { UsersPage } from "@/pages/UsersPage";
import { NewJobPage } from "@/pages/NewJobPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter basename={import.meta.env.BASE_URL}>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route element={<ProtectedRoute />}>
              <Route element={<Layout />}>
                <Route index element={<DashboardPage />} />
                <Route path="conversations" element={<ConversationsPage />} />
                <Route path="drift" element={<DriftAlertsPage />} />
                <Route path="experiments" element={<ExperimentsPage />} />
                <Route path="jobs/new" element={<NewJobPage />} />
                <Route path="jobs/:jobId" element={<JobDetailPage />} />
                <Route path="jobs" element={<JobsPage />} />
                <Route path="models" element={<ModelsPage />} />
                <Route path="datasets" element={<DatasetsPage />} />
                <Route path="users" element={<UsersPage />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}
