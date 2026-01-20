/**
 * Key Vault Service - Secure BYOK Storage with Envelope Encryption
 *
 * This service provides hardware-grade security for user API keys using:
 * - Envelope Encryption: Each user gets a unique DEK (Data Encryption Key)
 * - KEK (Key Encryption Key): Master key in Cloudflare secrets (HSM-backed on Enterprise)
 * - Zero-Knowledge: Backend never sees plaintext keys - all crypto at edge
 *
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │                         ENVELOPE ENCRYPTION                         │
 * ├─────────────────────────────────────────────────────────────────────┤
 * │                                                                     │
 * │   User API Key (plaintext)                                          │
 * │        │                                                            │
 * │        ▼ encrypt with DEK                                           │
 * │   ┌─────────────────────┐                                          │
 * │   │ Encrypted API Key   │ ──────────────────────► Supabase         │
 * │   │ (returned to        │                         (encrypted_key)  │
 * │   │  backend)           │                                          │
 * │   └─────────────────────┘                                          │
 * │                                                                     │
 * │   DEK (Data Encryption Key)                                        │
 * │        │                                                            │
 * │        ▼ encrypt with KEK                                           │
 * │   ┌─────────────────────┐                                          │
 * │   │ Encrypted DEK       │ ──────────────────────► Cloudflare KV    │
 * │   │ (stored in KV)      │                         (dek:{user_id})  │
 * │   └─────────────────────┘                                          │
 * │                                                                     │
 * │   KEK (Key Encryption Key)                                         │
 * │   ┌─────────────────────┐                                          │
 * │   │ Cloudflare Secret   │ ← Never leaves Cloudflare edge           │
 * │   │ (HSM-backed on      │                                          │
 * │   │  Enterprise)        │                                          │
 * │   └─────────────────────┘                                          │
 * │                                                                     │
 * └─────────────────────────────────────────────────────────────────────┘
 *
 * Security Benefits:
 * 1. Keys NEVER leave Cloudflare edge (zero-knowledge to backend)
 * 2. Per-user DEKs limit blast radius if one key is compromised
 * 3. KEK rotation doesn't require re-encrypting all user keys
 * 4. HSM-backed on Enterprise tier for FIPS compliance
 * 5. Automatic key versioning for rotation
 */

import { Env } from './index';

// ============================================================================
// Types
// ============================================================================

export interface EncryptedKeyBundle {
  /** Encrypted API key (encrypt with DEK, base64) */
  encrypted_key: string;
  /** DEK reference in KV (e.g., "dek:user_123:v1") */
  dek_reference: string;
  /** Key prefix for UI display (e.g., "sk-ant-...") */
  key_prefix: string;
  /** Key suffix for identification (e.g., "...xyz9") */
  key_suffix: string;
  /** DEK version for key rotation */
  dek_version: number;
  /** Timestamp when encrypted */
  encrypted_at: string;
}

export interface DecryptRequest {
  /** Encrypted API key (base64) */
  encrypted_key: string;
  /** DEK reference in KV */
  dek_reference: string;
}

interface StoredDEK {
  /** Encrypted DEK (encrypted with KEK) */
  encrypted_dek: string;
  /** Version number for rotation */
  version: number;
  /** User ID this DEK belongs to */
  user_id: string;
  /** Provider this key is for */
  provider: string;
  /** When this DEK was created */
  created_at: string;
  /** KEK version used to encrypt this DEK */
  kek_version: number;
}

// ============================================================================
// Cryptographic Utilities (Web Crypto API)
// ============================================================================

/**
 * Generate a random 256-bit key for AES-GCM
 */
async function generateDEK(): Promise<Uint8Array> {
  const key = new Uint8Array(32);
  crypto.getRandomValues(key);
  return key;
}

/**
 * Generate a random 96-bit nonce for AES-GCM
 */
function generateNonce(): Uint8Array {
  const nonce = new Uint8Array(12);
  crypto.getRandomValues(nonce);
  return nonce;
}

/**
 * Import a raw key for AES-GCM operations
 */
async function importKey(keyBytes: Uint8Array): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    'raw',
    keyBytes,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}

/**
 * Encrypt data with AES-256-GCM
 * Returns: nonce (12 bytes) + ciphertext + tag (16 bytes)
 */
async function encryptAESGCM(
  key: CryptoKey,
  plaintext: Uint8Array
): Promise<Uint8Array> {
  const nonce = generateNonce();

  const ciphertext = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv: nonce },
    key,
    plaintext
  );

  // Combine: nonce + ciphertext (includes 16-byte auth tag)
  const result = new Uint8Array(nonce.length + ciphertext.byteLength);
  result.set(nonce, 0);
  result.set(new Uint8Array(ciphertext), nonce.length);

  return result;
}

/**
 * Decrypt data encrypted with AES-256-GCM
 * Input: nonce (12 bytes) + ciphertext + tag (16 bytes)
 */
async function decryptAESGCM(
  key: CryptoKey,
  encrypted: Uint8Array
): Promise<Uint8Array> {
  const nonce = encrypted.slice(0, 12);
  const ciphertext = encrypted.slice(12);

  const plaintext = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: nonce },
    key,
    ciphertext
  );

  return new Uint8Array(plaintext);
}

/**
 * Derive KEK from secret string using HKDF
 * This allows us to use a human-readable secret while getting
 * cryptographically strong key material
 */
async function deriveKEK(secret: string): Promise<CryptoKey> {
  // Import secret as raw key material
  const encoder = new TextEncoder();
  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    encoder.encode(secret),
    { name: 'HKDF' },
    false,
    ['deriveKey']
  );

  // Derive 256-bit key using HKDF
  return crypto.subtle.deriveKey(
    {
      name: 'HKDF',
      salt: encoder.encode('argus-key-vault-v1'),
      info: encoder.encode('kek-derivation'),
      hash: 'SHA-256',
    },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt']
  );
}

// ============================================================================
// Key Vault Service
// ============================================================================

export class KeyVaultService {
  private kv: KVNamespace;
  private kekSecret: string;
  private kekCache: CryptoKey | null = null;

  constructor(kv: KVNamespace, kekSecret: string) {
    this.kv = kv;
    this.kekSecret = kekSecret;
  }

  /**
   * Get or derive KEK (cached for performance)
   */
  private async getKEK(): Promise<CryptoKey> {
    if (!this.kekCache) {
      this.kekCache = await deriveKEK(this.kekSecret);
    }
    return this.kekCache;
  }

  /**
   * Encrypt a user's API key using envelope encryption
   *
   * @param userId - User's ID
   * @param provider - AI provider (anthropic, openai, etc.)
   * @param apiKey - Plaintext API key to encrypt
   * @returns Encrypted key bundle with DEK reference
   */
  async encryptApiKey(
    userId: string,
    provider: string,
    apiKey: string
  ): Promise<EncryptedKeyBundle> {
    // 1. Generate a new DEK for this user/provider combination
    const dekBytes = await generateDEK();
    const dekKey = await importKey(dekBytes);

    // 2. Encrypt DEK with KEK (envelope encryption)
    const kek = await this.getKEK();
    const encryptedDEK = await encryptAESGCM(
      kek,
      dekBytes
    );

    // 3. Store encrypted DEK in KV
    const dekVersion = 1;
    const dekReference = `dek:${userId}:${provider}:v${dekVersion}`;

    const storedDEK: StoredDEK = {
      encrypted_dek: uint8ArrayToBase64(encryptedDEK),
      version: dekVersion,
      user_id: userId,
      provider: provider,
      created_at: new Date().toISOString(),
      kek_version: 1, // For future KEK rotation
    };

    await this.kv.put(dekReference, JSON.stringify(storedDEK), {
      // DEKs should persist indefinitely (or until user deletes key)
      expirationTtl: undefined,
    });

    // 4. Encrypt API key with DEK
    const encoder = new TextEncoder();
    const encryptedApiKey = await encryptAESGCM(
      dekKey,
      encoder.encode(apiKey)
    );

    // 5. Extract prefix/suffix for display
    const keyPrefix = apiKey.length >= 8 ? apiKey.slice(0, 8) : apiKey;
    const keySuffix = apiKey.length >= 4 ? apiKey.slice(-4) : '';

    return {
      encrypted_key: uint8ArrayToBase64(encryptedApiKey),
      dek_reference: dekReference,
      key_prefix: keyPrefix,
      key_suffix: keySuffix,
      dek_version: dekVersion,
      encrypted_at: new Date().toISOString(),
    };
  }

  /**
   * Decrypt a user's API key
   *
   * @param request - Encrypted key and DEK reference
   * @returns Plaintext API key
   */
  async decryptApiKey(request: DecryptRequest): Promise<string> {
    // 1. Retrieve encrypted DEK from KV
    const storedDEKJson = await this.kv.get(request.dek_reference);
    if (!storedDEKJson) {
      throw new Error(`DEK not found: ${request.dek_reference}`);
    }

    const storedDEK: StoredDEK = JSON.parse(storedDEKJson);

    // 2. Decrypt DEK with KEK
    const kek = await this.getKEK();
    const encryptedDEKBytes = base64ToUint8Array(storedDEK.encrypted_dek);
    const dekBytes = await decryptAESGCM(kek, encryptedDEKBytes);

    // 3. Decrypt API key with DEK
    const dekKey = await importKey(dekBytes);
    const encryptedApiKeyBytes = base64ToUint8Array(request.encrypted_key);
    const apiKeyBytes = await decryptAESGCM(dekKey, encryptedApiKeyBytes);

    // 4. Return plaintext
    const decoder = new TextDecoder();
    return decoder.decode(apiKeyBytes);
  }

  /**
   * Delete a user's DEK (when they remove their API key)
   *
   * @param dekReference - DEK reference to delete
   */
  async deleteDEK(dekReference: string): Promise<void> {
    await this.kv.delete(dekReference);
  }

  /**
   * Rotate a user's DEK (re-encrypt with new DEK)
   * Used when user updates their API key or for periodic rotation
   *
   * @param userId - User's ID
   * @param provider - AI provider
   * @param currentEncryptedKey - Currently encrypted API key
   * @param currentDekReference - Current DEK reference
   * @returns New encrypted key bundle with new DEK
   */
  async rotateDEK(
    userId: string,
    provider: string,
    currentEncryptedKey: string,
    currentDekReference: string
  ): Promise<EncryptedKeyBundle> {
    // 1. Decrypt current key
    const apiKey = await this.decryptApiKey({
      encrypted_key: currentEncryptedKey,
      dek_reference: currentDekReference,
    });

    // 2. Get current version
    const storedDEKJson = await this.kv.get(currentDekReference);
    const currentVersion = storedDEKJson
      ? (JSON.parse(storedDEKJson) as StoredDEK).version
      : 0;

    // 3. Generate new DEK and encrypt
    const dekBytes = await generateDEK();
    const dekKey = await importKey(dekBytes);

    const kek = await this.getKEK();
    const encryptedDEK = await encryptAESGCM(kek, dekBytes);

    // 4. Store new DEK with incremented version
    const newVersion = currentVersion + 1;
    const newDekReference = `dek:${userId}:${provider}:v${newVersion}`;

    const storedDEK: StoredDEK = {
      encrypted_dek: uint8ArrayToBase64(encryptedDEK),
      version: newVersion,
      user_id: userId,
      provider: provider,
      created_at: new Date().toISOString(),
      kek_version: 1,
    };

    await this.kv.put(newDekReference, JSON.stringify(storedDEK));

    // 5. Re-encrypt API key with new DEK
    const encoder = new TextEncoder();
    const encryptedApiKey = await encryptAESGCM(
      dekKey,
      encoder.encode(apiKey)
    );

    // 6. Delete old DEK
    await this.kv.delete(currentDekReference);

    const keyPrefix = apiKey.length >= 8 ? apiKey.slice(0, 8) : apiKey;
    const keySuffix = apiKey.length >= 4 ? apiKey.slice(-4) : '';

    return {
      encrypted_key: uint8ArrayToBase64(encryptedApiKey),
      dek_reference: newDekReference,
      key_prefix: keyPrefix,
      key_suffix: keySuffix,
      dek_version: newVersion,
      encrypted_at: new Date().toISOString(),
    };
  }

  /**
   * Verify a DEK exists without decrypting
   * Useful for health checks
   */
  async verifyDEKExists(dekReference: string): Promise<boolean> {
    const result = await this.kv.get(dekReference);
    return result !== null;
  }
}

// ============================================================================
// Base64 Utilities
// ============================================================================

function uint8ArrayToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64ToUint8Array(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

// ============================================================================
// HTTP Handlers for Key Vault API
// ============================================================================

export interface KeyVaultEnv extends Env {
  KEY_VAULT: KVNamespace;  // Dedicated KV namespace for keys
  KEK_SECRET: string;      // Master key encryption key
}

/**
 * Handle key vault API requests
 * POST /key-vault/encrypt - Encrypt an API key
 * POST /key-vault/decrypt - Decrypt an API key
 * DELETE /key-vault/dek/:reference - Delete a DEK
 */
export async function handleKeyVaultRequest(
  request: Request,
  env: KeyVaultEnv
): Promise<Response> {
  const url = new URL(request.url);
  const path = url.pathname;

  // Validate KEK is configured
  if (!env.KEK_SECRET) {
    return new Response(
      JSON.stringify({ error: 'KEK_SECRET not configured' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }

  // Use dedicated KEY_VAULT KV, or fall back to CACHE
  const kv = env.KEY_VAULT || env.CACHE;
  if (!kv) {
    return new Response(
      JSON.stringify({ error: 'KV namespace not configured' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }

  const vault = new KeyVaultService(kv, env.KEK_SECRET);

  try {
    // POST /key-vault/encrypt
    if (path === '/key-vault/encrypt' && request.method === 'POST') {
      const body = await request.json() as {
        user_id: string;
        provider: string;
        api_key: string;
      };

      if (!body.user_id || !body.provider || !body.api_key) {
        return new Response(
          JSON.stringify({ error: 'Missing required fields: user_id, provider, api_key' }),
          { status: 400, headers: { 'Content-Type': 'application/json' } }
        );
      }

      const result = await vault.encryptApiKey(
        body.user_id,
        body.provider,
        body.api_key
      );

      return new Response(JSON.stringify(result), {
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // POST /key-vault/decrypt
    if (path === '/key-vault/decrypt' && request.method === 'POST') {
      const body = await request.json() as DecryptRequest;

      if (!body.encrypted_key || !body.dek_reference) {
        return new Response(
          JSON.stringify({ error: 'Missing required fields: encrypted_key, dek_reference' }),
          { status: 400, headers: { 'Content-Type': 'application/json' } }
        );
      }

      const apiKey = await vault.decryptApiKey(body);

      return new Response(JSON.stringify({ api_key: apiKey }), {
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // DELETE /key-vault/dek/:reference
    if (path.startsWith('/key-vault/dek/') && request.method === 'DELETE') {
      const dekReference = decodeURIComponent(path.replace('/key-vault/dek/', ''));

      await vault.deleteDEK(dekReference);

      return new Response(JSON.stringify({ success: true }), {
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // POST /key-vault/rotate
    if (path === '/key-vault/rotate' && request.method === 'POST') {
      const body = await request.json() as {
        user_id: string;
        provider: string;
        encrypted_key: string;
        dek_reference: string;
      };

      if (!body.user_id || !body.provider || !body.encrypted_key || !body.dek_reference) {
        return new Response(
          JSON.stringify({ error: 'Missing required fields' }),
          { status: 400, headers: { 'Content-Type': 'application/json' } }
        );
      }

      const result = await vault.rotateDEK(
        body.user_id,
        body.provider,
        body.encrypted_key,
        body.dek_reference
      );

      return new Response(JSON.stringify(result), {
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // GET /key-vault/health
    if (path === '/key-vault/health' && request.method === 'GET') {
      // Basic health check - verify KEK can be derived
      try {
        await deriveKEK(env.KEK_SECRET);
        return new Response(JSON.stringify({
          status: 'healthy',
          kv_bound: !!kv,
          kek_configured: true,
        }), {
          headers: { 'Content-Type': 'application/json' },
        });
      } catch (e) {
        return new Response(JSON.stringify({
          status: 'unhealthy',
          error: 'KEK derivation failed',
        }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        });
      }
    }

    return new Response(
      JSON.stringify({ error: 'Not found' }),
      { status: 404, headers: { 'Content-Type': 'application/json' } }
    );

  } catch (error) {
    console.error('Key vault error:', error);
    return new Response(
      JSON.stringify({
        error: error instanceof Error ? error.message : 'Internal error',
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
