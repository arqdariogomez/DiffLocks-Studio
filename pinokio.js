
// Updated: 2025-12-15 01:33:18
const path = require('path')
const fs = require('fs')

module.exports = {
  title: "DiffLocks Studio",
  description: "AI-powered 3D hair generation from a single image",
  icon: "icon.png",
  menu: async (kernel) => {
    // Detectamos la instalación buscando la carpeta 'venv'
    let installed = await kernel.exists(__dirname, "venv")
    
    if (installed) {
      return [
        { html: '<i class="fa-solid fa-play"></i> Start', href: "start.json" },
        { html: '<i class="fa-solid fa-rotate"></i> Update', href: "update.json" },
        { html: '<i class="fa-solid fa-plug"></i> Re-install', href: "install.json" }
      ]
    } else {
      return [
        { html: '<i class="fa-solid fa-download"></i> Install', href: "install.json" }
      ]
    }
  }
}
