import type { Config } from 'tailwindcss'

export default {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  darkMode: 'class', // Use class-based dark mode (but we won't add 'dark' class, so it stays light)
  theme: {
    extend: {},
  },
  plugins: [],
} satisfies Config
