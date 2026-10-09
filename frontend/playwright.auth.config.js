import { defineConfig } from '@playwright/test'

// Same UI built with Supabase config, to exercise the sign-in gate. Supabase itself
// is never contacted: tests seed a session into localStorage and mock the API.
export default defineConfig({
  testDir: './e2e',
  testMatch: /auth\.spec\.js/,
  timeout: 30_000,
  use: { baseURL: 'http://localhost:4174' },
  webServer: {
    command: 'vite build --outDir dist-auth && vite preview --outDir dist-auth --port 4174 --strictPort',
    port: 4174,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: {
      VITE_SUPABASE_URL: 'http://localhost:54321',
      VITE_SUPABASE_ANON_KEY: 'test-anon-key',
      VITE_SUPABASE_OAUTH_PROVIDERS: 'github',
    },
  },
  projects: [{ name: 'desktop', use: { browserName: 'chromium' } }],
})
