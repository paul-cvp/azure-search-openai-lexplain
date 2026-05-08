import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['test/spec/**/*.js'],
    root: '.',
    assetsInclude: ['**/*.xml'],
    resolve: { alias: { stream: "stream-browserify" } },
    environment: 'happy-dom',
  },
});