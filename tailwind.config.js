/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./*.html", "./src/**/*.{jsx,js}"],
  theme: {
    extend: {
      colors: {
        'grammy-gold': '#D4AF37',
        'grammy-red': '#8B0000',
        'grammy-black': '#1A1A1A',
      },
      fontFamily: {
        'playfair': ['"Playfair Display"', 'serif'],
        'montserrat': ['Montserrat', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
