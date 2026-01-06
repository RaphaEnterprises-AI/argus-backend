'use client';

import { Sidebar } from './sidebar';
import { AppFooter } from './app-footer';

interface PageWrapperProps {
  children: React.ReactNode;
  showFooter?: boolean;
}

export function PageWrapper({ children, showFooter = true }: PageWrapperProps) {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 ml-64 flex flex-col">
        <main className="flex-1">
          {children}
        </main>
        {showFooter && <AppFooter />}
      </div>
    </div>
  );
}
