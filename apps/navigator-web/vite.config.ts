import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/platform-assets/',
  build: {
    manifest: true,
    rolldownOptions: { input: 'src/main.tsx' },
  },
});
