// Version configuration for Argus Dashboard
// This file exports version info that can be used throughout the app

import packageJson from '../package.json';

export interface VersionInfo {
  version: string;
  name: string;
  description: string;
  buildDate: string;
  environment: string;
}

export function getVersionInfo(): VersionInfo {
  return {
    version: packageJson.version,
    name: packageJson.name,
    description: packageJson.description,
    buildDate: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development',
  };
}

export const APP_VERSION = packageJson.version;
export const APP_NAME = 'Argus';
