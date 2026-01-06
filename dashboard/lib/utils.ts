import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
}

export function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(d);
}

export function getStatusColor(status: string): string {
  switch (status.toLowerCase()) {
    case 'passed':
      return 'text-green-500';
    case 'failed':
      return 'text-red-500';
    case 'running':
      return 'text-blue-500';
    case 'pending':
      return 'text-yellow-500';
    default:
      return 'text-muted-foreground';
  }
}

export function getStatusBgColor(status: string): string {
  switch (status.toLowerCase()) {
    case 'passed':
      return 'bg-green-500/10';
    case 'failed':
      return 'bg-red-500/10';
    case 'running':
      return 'bg-blue-500/10';
    case 'pending':
      return 'bg-yellow-500/10';
    default:
      return 'bg-muted';
  }
}
