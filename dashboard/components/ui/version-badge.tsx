'use client';

import { APP_VERSION, APP_NAME } from '@/lib/version';
import { cn } from '@/lib/utils';

interface VersionBadgeProps {
  className?: string;
  showName?: boolean;
  variant?: 'default' | 'minimal' | 'full';
}

export function VersionBadge({
  className,
  showName = false,
  variant = 'default'
}: VersionBadgeProps) {
  if (variant === 'minimal') {
    return (
      <span className={cn('text-xs text-muted-foreground', className)}>
        v{APP_VERSION}
      </span>
    );
  }

  if (variant === 'full') {
    return (
      <div className={cn('flex items-center gap-2', className)}>
        <span className="text-sm font-medium">{APP_NAME}</span>
        <span className="px-2 py-0.5 text-xs rounded-full bg-primary/10 text-primary font-mono">
          v{APP_VERSION}
        </span>
      </div>
    );
  }

  return (
    <span className={cn(
      'inline-flex items-center gap-1.5 px-2 py-1 text-xs rounded-md bg-muted text-muted-foreground font-mono',
      className
    )}>
      {showName && <span className="font-medium">{APP_NAME}</span>}
      <span>v{APP_VERSION}</span>
    </span>
  );
}
